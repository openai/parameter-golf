# SLOT-48 + VRL + QK-Gain 4.0 + XSA-11 — val_bpb 0.7271 (3-seed mean)

**val_bpb = 0.7271** (3-seed mean, std 0.0011) | 15.5-15.7 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | + SLOT BPB | Steps | Artifact |
|------|------------|------------|-------|----------|
| 1337 | 1.1248 | **0.7270** | 5698 | 15,649,297 |
| 42 | 1.1248 | **0.7260** | 5701 | 15,747,713 |
| 2025 | 1.1247 | **0.7282** | 5700 | 15,658,677 |
| **Mean** | **1.1248** | **0.7271** | | |

Beats PR #1321 (0.7406) by 0.014 BPB. Beats merged SOTA #1019 (1.1147) by 0.388 BPB.

## Architecture

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL (sigmoid-gated interpolation, init=-1.5), VE128 on layers 9,10, BigramHash(1024x128), XSA all 11 layers, QK-Gain 4.0, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997), SWA, Late QAT (threshold 0.15), int6 Hessian GPTQ + LZMA-9, Muon WD=0.04.

## SLOT-48 Details

- Per-sample hidden delta [bsz, 1, 512] + logit bias [bsz, 1, 1024]
- Scored-position masking (last stride=96 tokens per non-first window)
- 48 AdamW steps, cosine LR 0.012 -> 0.001
- Model weights frozen, delta optimized through detached hidden states
- Eval time: ~508s on 8xH100

## Compliance

- **Frozen-model SLOT**: model weights are never modified during evaluation. Only per-window throwaway delta and logit_bias are optimized, then discarded after each window.
- Same evaluation pattern as accepted PRs #1176, #1229, #1313.
- No n-gram cache, no eval-time GPTQ.
- AR self-gen GPTQ calibration (64 seqs, temp=0.8).
- Train time: 600s, SLOT eval time: ~508s (both within 10min budget).
