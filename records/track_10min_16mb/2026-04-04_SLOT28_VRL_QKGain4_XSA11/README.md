# SLOT-28 + VRL + QK-Gain 4.0 + XSA-11 — val_bpb 0.8275 (3-seed mean)

**val_bpb = 0.8275** (3-seed mean, std 0.0007) | 15.5-15.8 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | + SLOT BPB | Steps | Artifact |
|------|------------|------------|-------|----------|
| 1337 | 1.1249 | **0.8277** | 5700 | 15,641,357 |
| 42 | 1.1246 | **0.8267** | 5695 | 15,783,189 |
| 2025 | 1.1244 | **0.8281** | 5700 | 15,647,185 |
| **Mean** | **1.1246** | **0.8275** | | |

Beats PR #1313 (0.8637) by 0.036 BPB. Beats merged SOTA #1019 (1.1147) by 0.287 BPB.

## Key Differences vs PR #1313

| Parameter | PR #1313 | This PR |
|-----------|----------|---------|
| SLOT_STEPS | 24 | **28** |
| VRL | enabled, init=-1.5, interpolation | **enabled, init=-1.5, interpolation** |
| QK_GAIN_INIT | 4.0 | **4.0** |
| BigramHash | 1024x128 | **1024x128** |
| EVAL_STRIDE | 96 | **96** |
| SLOT_LR | 0.012 | **0.012** |
| SLOT_LR_MIN | 0.001 | **0.001** |

The primary improvement is increasing SLOT steps from 24 to 28, which provides additional eval-time optimization while staying within the 600s eval budget (~360s SLOT eval time).

## Architecture

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL (sigmoid-gated interpolation, init=-1.5), VE128 on layers 9,10, BigramHash(1024x128), XSA all 11 layers, QK-Gain 4.0, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997), SWA, Late QAT (threshold 0.15), int6 Hessian GPTQ + LZMA-9, Muon WD=0.04.

## SLOT-28 Details

- Per-sample hidden delta [bsz, 1, 512] + logit bias [bsz, 1, 1024]
- Scored-position masking (last stride=96 tokens per non-first window)
- 28 AdamW steps, cosine LR 0.012 -> 0.001
- Model weights frozen, delta optimized through detached hidden states
- Eval time: ~359s on 8xH100

## Compliance

- **Frozen-model SLOT**: model weights are never modified during evaluation. Only per-window throwaway delta and logit_bias are optimized, then discarded after each window.
- Same evaluation pattern as accepted PRs #1176, #1229, #1313.
- No n-gram cache, no eval-time GPTQ.
- AR self-gen GPTQ calibration (64 seqs, temp=0.8).
