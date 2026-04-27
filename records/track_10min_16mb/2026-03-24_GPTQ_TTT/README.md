# GPTQ Int6 + SGD Test-Time Training

## Summary

An 11-layer 512-dim GPT model trained with PR#414's 10-technique stack plus LeakyReLU(0.5)² activation, then improved at eval time with two post-training techniques:

1. **GPTQ int6 quantization** — Hessian-guided column-wise quantization that replaces naive per-row int6 rounding, reducing quantization error by 33.6% (Hessian-weighted MSE) and saving 0.0029 bpb.
2. **SGD test-time training (TTT)** — Continues training the model on validation data in a causal (score-first) manner, adapting the last 9 of 11 layers via SGD with cosine LR decay.

Combined A800 bpb: **1.1190** (estimated H100: ~1.122).

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Attention heads | 8 (4 KV, GQA) |
| MLP multiplier | 3.0× |
| Activation | LeakyReLU(0.5)² |
| Vocab size | 1024 (SentencePiece BPE) |
| Embeddings | Tied input/output |
| RoPE | Partial (16 dims), base 10000 |
| Logit softcap | 30.0 |

### Techniques from PR#414

- **XSA (Cross-Sequence Attention)**: Last 4 layers attend across batch sequences
- **EMA (Exponential Moving Average)**: Weight averaging for smoother convergence
- **U-Net skip connections**: Residual connections between early and late layers
- **SmearGate**: Learned gating for token mixing
- **BigramHash**: 2048-vocab bigram hash embeddings (dim=128) for local context
- **LNScale**: Learnable LayerNorm scaling
- **Value Embeddings (VE128)**: 128-dim value embeddings on layers 9-10
- **Late QAT**: Quantization-aware training enabled after loss reaches 0.15 threshold
- **SWA (Stochastic Weight Averaging)**: Checkpoint averaging every 50 steps

### Techniques we added

- **LeakyReLU(0.5)²**: Replaces ReLU² in MLP. Negative-slope 0.5 preserves gradient flow through the squaring operation. Saves 0.0026 bpb over ReLU² at zero compute cost.
- **GPTQ int6**: Post-training Hessian-guided quantization (Frantar et al., 2022; 256 calibration samples, block-128, percdamp=0.01). Saves 0.0029 bpb over naive int6 rounding.
- **SGD TTT**: Test-time training (Sun et al., 2024) with SGD (lr=0.002, momentum=0.9), cosine LR schedule (T_max=1893 chunks), freeze first 2 embedding/layer blocks, 3 epochs per 32K-token chunk, score-first causal evaluation. Saves ~0.0024 bpb (on GPTQ int6 baseline 1.1214).

### References

- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323.
- Sun, Y., Li, X., Dalal, K., Xu, J., Vikram, A., Zhang, G., Dubois, Y., Chen, X., Wang, X., Koyejo, S., Hashimoto, T., & Guestrin, C. (2024). Learning to (Learn at Test Time): RNNs with Expressive Hidden States. arXiv:2407.04620.

## Results

### Training (8× A800-SXM4-80GB, 1200s)

| Metric | Value |
|--------|-------|
| Training steps | 6202 / 20000 (wallclock capped) |
| Training time | 1200.0s (193.49 ms/step) |
| EMA val_bpb (pre-quant) | 1.1399 |
| Int6 roundtrip bpb | 1.1480 |
| Sliding-window bpb (stride=64) | 1.1243 |
| Artifact size (int6+zstd) | 15,871,987 bytes |

### Eval-Time Improvements (8× A800)

| Stage | bpb | Δ vs baseline | Time |
|-------|-----|---------------|------|
| Sliding window (stride=64) | 1.1243 | — | 162s |
| + GPTQ int6 | 1.1214 | −0.0029 | +19s |
| + SGD TTT (900 chunks) | **1.1190** | **−0.0053** | +546s |

### Artifact Compression

| Method | Size | Under 16MB? |
|--------|------|-------------|
| Int6 + zstd (default) | 15,871,987 | ✓ |
| GPTQ int6 + zstd-21 LDM | 15,750,888 | ✓ (249KB margin) |

The GPTQ model has higher entropy weights (Hessian-compensated), causing a slightly larger raw file. Long Distance Matching (LDM) in zstd exploits cross-layer weight pattern similarity, recovering the size difference.

### H100 Estimate

| Scenario | Estimated bpb |
|----------|---------------|
| Expected (eval-only delta +0.0005) | 1.1195 |
| Conservative (full A800→H100 delta +0.0036) | 1.1226 |
| Best case (more TTT chunks from faster H100) | 1.1175 |

H100 processes ~1490 chunks in 600s eval window (vs 900 on A800), which should further reduce bpb.

## TTT Configuration

```
SGD lr=0.002, momentum=0.9
Cosine LR schedule, T_max=1893 (total possible chunks)
freeze_blocks=2 (first 2 layers + embeddings frozen)
3 epochs per chunk
Chunk size: 32768 tokens
Score-first: score tokens before adapting on them (causal)
Eval stride: 64 (sliding window)
```

Temperature calibration was tested (T=0.94–1.05) but T=1.0 is optimal — the model is already well-calibrated after TTT.

## Technique Discovery Log

We ran 30+ experiments on 8×A800 to reach this configuration:

1. **Baseline reproduction** (1.2259 bpb) → confirmed A800/H100 correlation (+0.0015 delta)
2. **Core 5 stack** (1.1530) → int6 + MLP3x + sliding window + FP16 embed + zstd
3. **LeakyReLU²** (1.1509) → −0.0021 on Core 5
4. **PR#414 port** (1.1269) → 10-technique stack, best training-only result
5. **PR#414 + LeakyReLU²** (1.1243) → −0.0026 additive
6. **TTT: AdamW** (1.1305–1.1335) → worse than SGD, adaptive LR causes drift
7. **TTT: SGD cosine** (1.1238) → −0.0031, cosine > constant LR
8. **TTT: 10 epochs** (1.1310) → worse, catastrophic forgetting erases gains
9. **TTT: Sidecar** (1.1282–1.1290) → requires co-training, random init fails
10. **TTT: LoRA** (1.1243–1.1576) → model too small for rank-16 (3% subspace)
11. **GPTQ** (1.1214) → −0.0029 vs naive int6
12. **GPTQ + TTT** (**1.1190**) → best result, gains additive
13. **Stride tuning** (inconclusive) → Swept stride {32, 64, 128, 256, 512} but wrong model checkpoint loaded during sweep (T045). Relative differences <0.002 bpb suggest stride is a marginal lever. Default stride=64 retained.

Dead ends: AdamW TTT, 10-epoch TTT, sidecar TTT from random init, LoRA TTT on 512-dim model, temperature calibration, stride tuning.

## Reproduction

### Training
```bash
# On 8xH100 or 8xA800
torchrun --nproc_per_node=8 train_gpt.py
```

### GPTQ Quantization (post-training)
```bash
# Requires final_model.pt from training
python eval_gptq.py
```

### TTT Evaluation
```bash
# Uses GPTQ-quantized model
# TTT config is controlled by env vars:
TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=2 \
TTT_CHUNK_TOKENS=32768 TTT_MAX_CHUNKS=900 \
TTT_SKIP_BASELINE=1 TTT_LR_SCHEDULE=cosine \
torchrun --nproc_per_node=8 eval_ttt.py
```

## Hardware & Environment

- Training: 8× A800-SXM4-80GB (1200s wallclock)
- Eval: 8× A800-SXM4-80GB (GPTQ: 19s, TTT: 546s)
- PyTorch 2.8.0+cu129
- FlashAttention 2.8.3
- CUDA 12.9 (A800), target CUDA 12.8 (H100 competition env)

## Artifact Contents

- `train_gpt.py` — training script (LeakyReLU² modification of PR#414)
- `train.log` — full training output (6202 steps, 1200s)
- `submission.json` — structured metadata
- `gptq_results.json` — GPTQ quantization metrics
- `ttt_gptq_results.json` — TTT evaluation metrics
- `README.md` — this file

## Limitations

- **Single seed**: Only 1 A800 training run. Competition requires 3-seed H100 validation.
- **No H100 run yet**: bpb estimate is projected from A800 results.
- **TTT coverage**: 900/1893 chunks processed on A800 (47.5%). H100 should reach ~79% (1490 chunks).
- **A800 vs H100 gap**: Training produces fewer steps on A800 (6202) vs H100 (~13000+), so the A800-trained model is weaker. Final submission must train on H100.
