# QAT + Neural Cache + LoRA TTT (Non-Record Submission)

**val_bpb: 1.4245** (sliding window, post int5/int6+zstd quantization roundtrip, 1 seed)

This is a non-record submission exploring three eval-time techniques stacked on the current #1 training recipe. The QAT implementation has a bug (quantization penalty is ~0.25 BPB instead of expected ~0.02), making this run non-competitive. Submitting for transparency and to document the approach for iteration.

## Approach

Built on PR by @thwu1 (Int5-MLP + BigramHash + SWA), adding:

### 1. Quantization-Aware Training (QAT)
STE fake-quantization during training: int5 (clip=15) for MLP layers, int6 (clip=31) for attention. The model learns to be robust to quantization noise. **Bug found:** The STE uses symmetric clipping while the export uses percentile-based per-row scaling — this mismatch caused the model to optimize for the wrong quantization target, resulting in a 0.25 BPB penalty instead of the expected ~0.02.

### 2. Neural Cache
During sliding window eval, maintain a ring buffer of pre-lm_head hidden states (dim=512, bf16). For each token, compute cosine similarity against cached states, build a cache distribution via softmax-weighted scatter, and interpolate with model predictions using logaddexp. Causal token-by-token scoring with document boundary resets prevents information leakage.

### 3. LoRA Test-Time Training
Per-document rank-8 LoRA adaptation on lm_head, Q, and V projections during evaluation. Documents batched (batch_size=64), chunks scored before training (no leakage), with entropy-gated updates.

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (1536 hidden)
- BigramHash(10240, dim=128), SmearGate, orthogonal init
- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99
- SWA: last 40% of warmdown, every 50 steps, 24 checkpoints averaged
- seq_len=2048, batch=786K tokens

## Results
| Seed | Pre-quant val_bpb | Post-quant sliding val_bpb | Steps | Artifact |
|------|-------------------|---------------------------|-------|----------|
| 1337 | 1.1739 | 1.4245 | 5109 | 15.77 MB |

## Known Issues
1. **QAT mismatch:** STE clip ranges don't match export quantization format — needs per-row percentile clipping in the STE to match `quantize_intN_per_row`
2. **Pre-quant BPB already worse than SOTA:** 1.1739 vs 1.1428 — QAT may be hurting convergence with current hyperparameters
3. Only 1 seed (need 3+ for statistical significance)

## Next Steps
- Run without QAT to verify base recipe reproduces 1.1428
- Fix QAT to match exact export quantization format
- Run neural cache + TTT eval on a working checkpoint
- Sweep cache hyperparameters (theta, lambda)

## Command
```bash
RUN_ID=run1_seed1337 SEED=1337 QAT_ENABLED=1 EVAL_STRIDE=64 EVAL_STRATEGY=combined \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
