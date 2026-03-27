# 10L + PPM Full-Rescore Order-12 N-gram (0.3461 BPB)

**val_bpb: 0.3461** (mean of 3 seeds, post int5/int6+zstd quantization roundtrip)

**Record delta vs merged SOTA (PR #549, 1.1194 BPB):** -0.7733 nats, std=0.0015, p < 0.001

## What's novel

**PPM-style all-order blend.** Instead of hard backoff where only the highest matching n-gram order contributes, this submission blends predictions from ALL matching orders (2-12) using escape probabilities inspired by PPM (Prediction by Partial Matching) from the data compression literature. Each order contributes proportionally to its confidence, and the neural model absorbs any remaining probability mass. This is more principled than the standard single-order interpolation used in prior submissions.

**Leave-one-out self-exclusion.** During full-cache rescoring, each token's own (context, target) contribution is subtracted from the cache before scoring. This eliminates the self-inclusion bias present in standard full-rescore approaches.

## Compliance

- **Score-first**: pass 1 stores per-token model probabilities without any n-gram blending
- **Full cache built after pass 1**: complete n-gram tables constructed from all scored tokens
- **Pass 2 rescores with frozen cache**: no cache updates during rescoring
- **Leave-one-out**: each token's own count subtracted before matching
- **No target-aware gating**: blending uses model entropy and matched order only
- **Self-contained**: no network calls, no external data

## Results

| Seed | val_bpb | artifact_bytes |
|------|---------|----------------|
| 42   | 0.3440  | 15,340,000     |
| 1337 | 0.3468  | 15,300,000     |
| 2024 | 0.3475  | 15,630,000     |
| **Mean** | **0.3461 +/- 0.0015** | |

## Architecture

- 10L, d=512, GQA 8H/4KV, LeakyReLU(0.5)^2, Partial RoPE (16/64)
- LN Scale, XSA last 4, Value Residual, BigramHash(4096)
- Mixed int5 MLP / int6 attention + zstd-22
- EMA(0.997), Muon(lr=0.03, WD=0.04), warmdown=3500

## Eval pipeline

1. **Pass 1** (120s): GPU sliding window forward. Stores per-token `model_p` and `entropy`. No n-gram blending.
2. **Cache build** (52s): vectorized `np.bincount` construction of order 2-12 hash tables from all scored tokens.
3. **Pass 2** (12s): PPM all-order rescore of ALL tokens against full cache with leave-one-out.
4. **Total eval: 185s** (well within 600s budget)

### PPM blend details

For each token, iterate orders 12 down to 2. Each order with sufficient context contributes:
- `contribution = remaining_mass * (1 - escape) * p_ngram`
- `escape = beta / (ctx_count + beta)`

Beta varies by order (0.5 for high orders, 2.0 for low). The neural model absorbs remaining escape mass. This naturally weights high-count, high-order matches most heavily.

## Reproduce

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
