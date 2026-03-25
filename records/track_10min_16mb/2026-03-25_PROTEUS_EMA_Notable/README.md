# PROTEUS EMA — Notable Non-Record Submission

**val_bpb:** 1.1836 (3-seed mean, std 0.0005)
**Status:** Notable non-record (does not beat SOTA)

## Architecture

Baseline 9-layer transformer + EMA weight averaging. No architectural changes — 26 lines added to the baseline `train_gpt.py`.

| Parameter | Value |
|-----------|-------|
| Layers | 9 |
| Dimension | 512 |
| Heads | 8 (4 KV, GQA) |
| MLP | 3x (1536) |
| Vocab | 1024 BPE, tied embeddings |
| EMA decay | 0.995, every step |
| Quantization | INT8 + zlib |
| Artifact size | 15.88 MB |

## 3-Seed Results (8×H100 SXM)

| Seed | val_bpb | Steps | Artifact (bytes) |
|------|---------|-------|-------------------|
| 42   | 1.1836  | 11,876 | 15,878,748 |
| 1337 | 1.1841  | 11,871 | 15,869,197 |
| 2024 | 1.1831  | 11,875 | 15,878,741 |
| **Mean** | **1.1836** | — | **std: 0.0005** |

## Documented Negative Results

This submission's primary value is the documented negative results:

1. **INT4 post-training quantization fails catastrophically** — roundtrip BPB goes from 1.44 to 3.73. Per-row, per-group (gs=64), and QAT with STE all fail. Root cause: quantization error compounds through layers (cosine similarity drops to 0.90 at 18 layers).

2. **Shared-weight depth recurrence (LoopFormer) loses to more tokens** at this training budget — 1-pass (9 effective layers, 6.5B tokens) beats 2-pass (18 effective layers, 3.6B tokens) by 0.019 BPB.

3. **EMA reduces quantization gap** from 0.0072 to 0.0048 BPB by smoothing weight distributions, but the training loss improvement is marginal.

## Logs

- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2024.log`
