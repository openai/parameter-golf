# SLOT-48 — val_bpb 0.7406 (3-seed mean)

**val_bpb = 0.7406** (3-seed mean, std 0.0051) | 15.75-15.82 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | + SLOT BPB | Steps | Artifact |
|------|------------|------------|-------|----------|
| 1337 | 1.126 | **0.7450** | 6578 | 15,815,983 |
| 42 | 1.121 | **0.7350** | 6576 | 15,751,595 |
| 2024 | 1.122 | **0.7416** | 6588 | 15,793,375 |
| **Mean** | **1.123** | **0.7406** | | |

Beats PR #1313 (0.8637) by 0.123 BPB. Beats best pending (#1229, 0.9300) by 0.190 BPB.

## What Changed vs PR #1313

Only SLOT step count — same model, same training, same LR, same stride:

| Parameter | PR #1313 | This PR |
|-----------|----------|---------|
| SLOT_STEPS | 24 | **48** |

## Architecture

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL, VE128, BigramHash(1024), XSA all 11 layers, QK-Gain 4.0, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997), Late QAT, int6+lzma, FA3 Hopper, Muon WD=0.04.

## SLOT-48 Details

- Per-sample hidden delta [bsz, 1, 512] + logit bias [bsz, 1, 1024]
- Scored-position masking (last stride=96 tokens per non-first window)
- 48 AdamW steps, cosine LR 0.012 -> 0.001
- Model weights frozen, delta optimized through detached hidden states
- Eval time: ~409s on 8xH100 (under 10-min eval budget)

## SLOT Scaling Behavior

| Steps | BPB (seed 1337) | Delta |
|-------|-----------------|-------|
| 16 | 0.949 | baseline |
| 24 | 0.868 | -0.081 |
| **48** | **0.745** | **-0.123** |

SLOT continues to improve well beyond the 24-32 step range. No sign of convergence at 48 steps.

## Compliance

- **Frozen-model SLOT**: model weights are never modified during evaluation. Only per-window throwaway delta and logit_bias parameters are optimized, then discarded. Same evaluation pattern as accepted PRs #1176 and #1229.
- No n-gram cache, no eval-time GPTQ
- Self-contained, no network calls
- All seeds within time and size budgets

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training: ~600s. Eval: ~409s. Total: ~17 min.

## Credits

- Base: PR #175, PR #1303, PR #1313 (@anthony-maio)
- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag), PR #1229 (@resouer)
- QK-Gain 4.0: PR #1125
- XSA: PR #1176 (@bigbag)
- VRL: ResFormer (arXiv:2410.17897)
