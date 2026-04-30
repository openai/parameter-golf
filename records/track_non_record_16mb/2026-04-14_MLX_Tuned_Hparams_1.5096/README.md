# Tuned Hyperparameters for MLX Baseline — 1.5096 BPB Locally

**Author:** @seekerPrice
**Date:** 2026-04-14
**Track:** non_record_16mb (H100 validation pending)
**Hardware:** Apple M5 MacBook Pro (MLX framework)

## TL;DR

A/B comparison at **same model, same training config, different hyperparameters**:

| Experiment | Matrix LR | Muon Momentum | QK-Gain | val_bpb |
|-----------|-----------|---------------|---------|---------|
| EXP-042 (SOTA defaults) | 0.022 | 0.99 | 5.25 | 1.5596 |
| **EXP-048 (tuned)** | **0.02** | **0.95** | **4.0** | **1.5096** |
| | | | **Δ:** | **-0.0500** |

**Pure hyperparameter tuning gave -0.05 BPB** at 5000-step MLX training scale.

## Changes

Only 4 hyperparameters changed. Same architecture (11L × 512d × 4xMLP, depth recurrence L3,4,5, parallel residuals L7+, SP4096 casefold tokenizer, Muon + AdamW split optimizer).

```diff
-matrix_lr = 0.022
+matrix_lr = 0.02

-muon_momentum = 0.99
+muon_momentum = 0.95

-muon_momentum_warmup_start = 0.95
+muon_momentum_warmup_start = 0.90

-qk_gain_init = 5.25
+qk_gain_init = 4.0
```

## Why These Values

**Starting from SOTA defaults** (tuned for H100 large-batch training), we hypothesized they might be too aggressive for our small-batch MLX runs:

- **Matrix LR 0.022 → 0.02**: less aggressive update magnitude at smaller batch (8K tokens vs SOTA's 524K)
- **Muon momentum 0.99 → 0.95**: less backward-looking; helps in small-batch noisy gradient regime
- **Muon momentum warmup 0.95 → 0.90**: slower warmup reduces early training spikes
- **QK-Gain 5.25 → 4.0**: softer attention. Pairs well with Partial RoPE (only 16/64 dims rotated) — too-sharp attention overreacts to the non-rotated content dimensions

## Experimental Setup

**Training:** 5000 steps, 8K tokens/step = 40M total tokens (vs SOTA H100: 2.4B)
**Validation:** Full FineWeb val split (47.7M tokens, 524K batch)
**Model:** 33.8M params, ~12.65 MB artifact after int6+Brotli compression

**Environment variables (EXP-048 winning config):**
```bash
export MATRIX_LR=0.02
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.90
export QK_GAIN_INIT=4.0
# Other (unchanged from SOTA):
export TIED_EMBED_LR=0.03
export SCALAR_LR=0.02
export GRAD_CLIP_NORM=0.3
export MUON_MOMENTUM_WARMUP_STEPS=60
```

## Key Caveat

**This is a LOCAL MLX result at 40M tokens — not a H100 competition submission.**

The SOTA leaderboard results are at 2.4B tokens on 8×H100. Our 40M-token result isn't directly comparable. However, the **A/B improvement within our framework** (-0.05 BPB from hyperparameter tuning alone) should transfer to larger scales — tuned hyperparameters are generally scale-stable for small deltas like these.

**Prediction:** At H100 scale, these tuned values should give ~0.01-0.02 BPB improvement over SOTA-default hyperparameters, all else equal.

## Why Share This

Small improvements accumulate. If these tuned hyperparameters give even -0.005 BPB at H100 scale, that's meaningful against the 1.0810 leaderboard SOTA. Sharing empirical evidence helps the community.

## Methodology Notes

**3-AI collaboration** (Claude + Gemini + Codex) independently recommended these exact hyperparameters based on theoretical analysis (Muon at large momentum is unstable with small batch; QK-Gain 5.25 over-sharpens with partial RoPE). We then validated empirically.

## Status

- [x] Local MLX A/B test (5000 steps) — EXP-042 vs EXP-048
- [x] Documented in project findings
- [ ] H100 3-seed validation (pending compute credits)
- [ ] Combined with SOTA architecture stack on H100

## Related PRs

- #1595 (open): Previous non-record submission (3x MLP + QAT) — superseded by this result
- Applying for H100 credits at: https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf

## Attribution

Baseline SOTA hyperparameters sourced from PR #1394 (clarkkev), #1437 (dexhunter), #1412 (Robby955), #1445 (X-Abhishek-X). Our contribution is the specific re-tuning for the 8K-batch MLX regime and empirical validation.
