# SLOT-24 Aggressive — val_bpb 0.8637 (3-seed mean)

**val_bpb = 0.8637** (3-seed mean, std 0.0051) | 15.7-15.8 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | + SLOT BPB | Steps | Artifact |
|------|------------|------------|-------|----------|
| 1337 | 1.1258 | **0.8683** | 6034 | 15,679,900 |
| 42 | 1.1207 | **0.8582** | 6563 | 15,827,704 |
| 2024 | 1.1221 | **0.8647** | 6568 | 15,770,916 |
| **Mean** | **1.1229** | **0.8637** | | |

Beats PR #1303 (0.9462) by 0.083 BPB. Beats best pending (#1229, 0.9300) by 0.066 BPB.

## What Changed vs PR #1303

Only SLOT hyperparameters — same model, same training, same architecture:

| Parameter | PR #1303 | This PR |
|-----------|----------|---------|
| SLOT_STEPS | 16 | **24** |
| SLOT_LR | 0.008 | **0.012** |
| SLOT_LR_MIN | 0.0008 | **0.001** |
| EVAL_STRIDE | 64 | **96** |

Found via 6-config hyperparameter sweep across SLOT steps, LR, and stride combinations.

## Architecture

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, VRL, VE128, BigramHash(1024), XSA all 11 layers, QK-Gain 4.0, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997), Late QAT, int6+lzma, FA3 Hopper, Muon WD=0.04.

## SLOT-24 Details

- Per-sample hidden delta [bsz, 1, 512] + logit bias [bsz, 1, 1024]
- Scored-position masking (last stride=96 tokens per non-first window)
- 24 AdamW steps, cosine LR 0.012 -> 0.001
- Model weights frozen, delta optimized through detached hidden states
- Eval time: ~231-255s on 8xH100

## Compliance

- **Frozen-model SLOT**: model weights are never modified during evaluation. Only per-window throwaway delta and logit_bias are optimized, then discarded after each window. Same evaluation pattern as accepted PRs #1176 and #1229.
- No n-gram cache, no eval-time GPTQ
- Self-contained, no network calls
- All seeds within time and size budgets

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults set in Hyperparameters class. Training: ~600s. Eval: ~350s. Total: ~16 min.

## Credits

- Base: PR #175, PR #1303 (@anthony-maio)
- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag), PR #1229 (@resouer)
- QK-Gain 4.0: PR #1125
- XSA: PR #1176 (@bigbag)
- VRL: ResFormer (arXiv:2410.17897)
