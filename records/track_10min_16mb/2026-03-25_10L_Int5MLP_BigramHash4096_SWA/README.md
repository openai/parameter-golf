# 10L Int5-MLP + Multi-Order N-gram Backoff (0.9123 BPB)

**val_bpb: 0.9123** (mean of 3 seeds, post int5/int6+zstd quantization roundtrip)

## Results

| Seed | val_bpb | artifact_bytes |
|------|---------|----------------|
| 42   | 0.9128  | 15,320,000     |
| 1337 | 0.9121  | 15,630,000     |
| 2024 | 0.9121  | 15,330,000     |
| **Mean** | **0.9123 +/- 0.0003** | |

## Architecture

- 10 layers, d=512, 8 heads, 4 KV heads (GQA)
- MLP: 3x expansion (1536), LeakyReLU(0.5)^2 activation
- BigramHash: 4096 buckets, 128-dim projection
- SmearGate, U-Net skip connections
- Partial RoPE (16/64 dims), LN Scale (1/sqrt(L+1))
- XSA on last 4 layers, Value Residual (layer-0 V blend)
- Tied embeddings, logit softcap=30.0

## Training

- Muon optimizer (matrices) + AdamW (embeddings/scalars), WD=0.04
- EMA: decay=0.997, updated every 10 steps on GPU
- Warmdown: 3500 steps, warmup: 5 steps
- Wallclock cap: 600s on 8xH100 (~6020 steps)
- val_loss_every=0 to maximize training steps

## Quantization

- Int5 per-row for MLP weights, Int6 per-row for attention
- FP16 passthrough for small/control tensors
- Magnitude pruning (3% threshold) before quantization
- zstd-22 compression

## Evaluation: Multi-Order N-gram Backoff

Legal score-first hashed n-gram cache with entropy-adaptive interpolation:

- Orders 2 through 7 with backoff (highest matching order wins)
- Separate hash tables per order (4M buckets each, uint32 counts)
- Entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
  - Low model entropy (confident): alpha near 0.05, trust model
  - High model entropy (uncertain): alpha near 0.60, trust n-gram
- Score-first: cache updated only AFTER segment scoring
- Sliding window stride=64, eval_batch_seqs=64
- Eval time: ~163s on 8xH100 (well within 10-min budget)

## Based on

- thwu1's 10L Int5-MLP architecture (base model)
- PR #727 (multi-order n-gram backoff concept)
- PR #549 (LeakyReLU^2 + score-first TTT)
- PR #287 (XSA, EMA, Partial RoPE, LN Scale)

## Reproduce

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Disable n-gram cache (base model only):
```bash
NGRAM_EVAL_ORDER=0 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
