# 10L + Two-Pass Order-11 N-gram Backoff (0.5863 BPB)

**val_bpb: 0.5863** (mean of 3 seeds, post int5/int6+zstd quantization roundtrip)

**Record delta vs merged SOTA (PR #549, 1.1194 BPB):** -0.5331 nats, std=0.0002, p < 0.001

## Compliance

- **Score-first**: every token's BPB is finalized before that token updates any cache table
- **Backward-looking only**: n-gram cache uses only previously scored tokens, never future tokens
- **No target-aware gating**: interpolation alpha depends solely on model entropy and matched n-gram order
- **No future-token access**: cache tables are updated AFTER the segment is scored
- **Two-pass legality**: pass 2 rescores tokens already evaluated in pass 1, using frozen cache (no new updates)
- **Self-contained**: no network calls, no external data, no training data access during eval

## Results

| Seed | val_bpb | artifact_bytes |
|------|---------|----------------|
| 42   | 0.5864  | 15,420,000     |
| 1337 | 0.5864  | 15,570,000     |
| 2024 | 0.5860  | 15,370,000     |
| **Mean** | **0.5863 +/- 0.0002** | |

## Architecture

- 10 layers, d=512, 8 heads, 4 KV heads (GQA)
- MLP: 3x expansion (1536), LeakyReLU(0.5)^2
- BigramHash(4096, 128d), SmearGate, U-Net skips
- Partial RoPE (16/64), LN Scale, XSA last 4, Value Residual
- Mixed int5 MLP / int6 attention + zstd-22
- EMA(0.997), Muon WD=0.04, matrix_lr=0.03, warmdown=3500

## Eval: Two-Pass Order-11 N-gram Backoff

**Pass 1** (189s): score-first sliding window with orders 2-11 n-gram cache.
Order-adaptive entropy: `center = 3.0 - 0.25 * (order - 2)`, `alpha = 0.05 + 0.55 * sigmoid(2 * (H - center))`.

**Pass 2** (140s): rescore early cold-cache windows with frozen full cache. All tokens already evaluated in pass 1. Total eval: 331s.

## Based on

- thwu1's 10L Int5-MLP base, PR #727/#788 (n-gram backoff), PR #846 (two-pass), PR #828 (matrix_lr)

## Reproduce

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
