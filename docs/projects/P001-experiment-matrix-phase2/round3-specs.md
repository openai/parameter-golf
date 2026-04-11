# Round 3: Formal Specifications

## R3-1: GPTQ int6 (Full Hessian)

### Mathematical Spec
Post-training quantization using GPTQ algorithm (Frantar et al. 2022):
```
For each weight matrix W ∈ R^{m×n}:
  1. Collect Hessian H = X^T X from calibration data        (X = layer input activations)
  2. Damp: H += 0.01 × mean(diag(H)) × I                   (regularize for invertibility)
  3. Permute columns by descending diag(H)                   (most-activating first)
  4. Compute H_inv via double Cholesky: L = cholesky(H), H_inv = cholesky(cholesky_inverse(L), upper=True)
  5. For each column block [i1..i2] of size 128:
     For each column i in block:
       q_i = clamp(round(w_i / scale), -clip, clip)         (quantize to int6, clip=31)
       err_i = (w_i - q_i × scale) / H_inv[i,i]            (scaled error)
       W[:, i+1:] -= err_i × H_inv[i, i+1:]                (intra-block compensation)
     W[:, i2:] -= Err_block × H_inv[i1:i2, i2:]            (inter-block compensation)
```

### Calibration Data: AR Self-Generation
```
Generate calibration sequences autoregressively from the trained model:
  - 64 sequences × 2048 tokens each
  - Temperature = 0.8
  - No access to training or validation data (competition-legal)
```

### Clip Range Selection
Sweep 5 percentile candidates (99.90%, 99.93%, 99.95%, 99.97%, 100%) per weight matrix. Select the clip range that minimizes roundtrip quantization error.

### Interface
- **New functions**: `quantize_int6_gptq(weight, hessian, clip_range=31, block_size=128)`
- **New function**: `collect_hessian(model, calib_data)` — register hooks, run calibration
- **New function**: `generate_calibration_data(model, num_seqs=64, seq_len=2048, temp=0.8)`
- **Integration**: Post-training, before model serialization
- **Env var**: `GPTQ=1`, `GPTQ_CALIB_BATCHES=64`

### Parameter Budget
int6 quantization: 6 bits per weight + per-group scale factor.
- At block_size=128: each 128-weight group gets one fp16 scale (2 bytes)
- Storage per weight: 6/8 + 2/128 = 0.766 bytes (vs 2 bytes for fp16)
- Compression ratio: ~2.6x before zlib

### Verifiable DoD
1. Quantized model produces valid logits (no NaN/inf)
2. `val_bpb(quantized)` is within 0.005 of `val_bpb(float)` (low roundtrip penalty)
3. `final_model.int8.ptz` < 16MB (competition limit)
4. Calibration data is self-generated (no train/val data access)
5. Hessian is symmetric positive definite after damping

### References
- Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" (arXiv 2210.17323, 2022)
- AR self-generation calibration: parameter-golf [PR #1019](https://github.com/openai/parameter-golf/pull/1019)

---

## R3-2: Sliding Window Evaluation (stride=64)

### Mathematical Spec
```
Standard eval:    Split val set into non-overlapping chunks of length T.
                  Each chunk is scored independently. Positions at start of each
                  chunk have minimal context → worse predictions.

Sliding window:   Evaluate with stride S < T. Each position (except first S)
                  is scored with at least (T - S) tokens of context.
                  
For stride=64, T=1024:
  Window 0: score positions [0..1023]      (first 64 positions have limited context)
  Window 1: score positions [64..1087]     (positions 64-127 now have 960 tokens of context)
  ...
  Only score the LAST S=64 positions of each window (except the first window).
```

### Interface
- **Modified eval function**: `evaluate_sliding(model, data, stride=64, seq_len=1024)`
- **Env var**: `EVAL_STRIDE=64`

### Computational Cost
Eval takes `T/S` times longer. At stride=64, T=1024: 16x more eval forward passes.
For val set of 62M tokens: ~16x standard eval time. Budget ~5-8 minutes.

### Verifiable DoD
1. `val_bpb(sliding) <= val_bpb(standard)` — sliding always improves or equals
2. Expected improvement: ~0.02-0.03 bpb based on SOTA submissions
3. All val tokens are scored exactly once
4. No data leakage — only past tokens are visible (causal mask)

### References
- Sliding window evaluation: parameter-golf leaderboard standard since [PR #198](https://github.com/openai/parameter-golf/pull/198)
- Press et al. "Train Short, Test Long" (2021) — context extension via sliding windows

---

## R3-3: Legal Score-First TTT

### Mathematical Spec
```
Split val set into N non-overlapping chunks of size C.

For chunk n = 0, 1, ..., N-1:
  1. SCORE phase (inference_mode, no weight changes):
     - Run model on chunk_n with sliding window eval
     - Accumulate scored loss into final metric
     - The model used for scoring has been fine-tuned only on chunks 0..n-1
     
  2. TRAIN phase (SGD fine-tuning):
     - Fine-tune model on chunk_n using SGD
     - lr = lr_base × 0.5 × (1 + cos(π × n / N))    (cosine decay across chunks)
     - SGD with momentum=0.9, grad_clip=1.0
     - 3 epochs over chunk_n
     - Freeze first K layers (prevent catastrophic forgetting)
     - All-reduce gradients across workers

Legality invariant: chunk_n is SCORED before TRAINED on.
```

### Interface
- **New function**: `legal_ttt(model, val_data, lr=0.002, epochs=3, freeze_blocks=2)`
- **Replaces**: Standard eval at the end of training
- **Env var**: `TTT=1`, `TTT_LR=0.002`, `TTT_EPOCHS=3`, `TTT_FREEZE_BLOCKS=2`

### Computational Cost
- Scoring: same as sliding window eval
- Training: 3 epochs × N chunks × forward+backward
- Total: ~400-500 seconds (must fit within 600s evaluation budget)

### Verifiable DoD
1. `val_bpb(TTT) < val_bpb(no-TTT)` — TTT should improve
2. Expected improvement: ~0.0025 bpb based on SOTA #2
3. **Legality check**: For each chunk, the model weights used for scoring were NOT trained on that chunk
4. Frozen blocks: first K layers have zero gradient during TTT
5. Total wall time (train + eval) < 1200 seconds (600s train + 600s eval budget)

### References
- Legal Score-First TTT: parameter-golf [PR #549](https://github.com/openai/parameter-golf/pull/549)
- Test-Time Training: Sun et al. "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" (ICML 2020)
- End-to-end TTT: [test-time-training.github.io](https://test-time-training.github.io/e2e.pdf) (2025)
