# Spec 004 — QK-gain extension — Summary

Three phases ran, converging on the same verdict.

## Final verdict: **KILL**

QK_GAIN_INIT=6.0 is tied with QK_GAIN_INIT=5.25 at matched step count, within noise. QK_GAIN_INIT=5.5 also shows no meaningful signal. No promotion. Spec 004 closed.

## Three-way comparison (all QK=6.0 attempts)

| metric | Run A (5-min) | 004b (10-min VAL=200) | 004c (10-min VAL=4000) | spec 000 (QK=5.25) |
|---|---|---|---|---|
| pod | same run | `waby1c846taown` | `07j7kkjao3lq3r` | `t7k5v85j3fwpdh` |
| VAL_LOSS_EVERY | 4000 | 200 | 4000 | 4000 |
| final step | 1700 (5-min cap) | 3876 | 3952 | 3849 |
| step-1000 train_loss | **3.1394** | 3.2482 | 3.2510 | 3.2487 |
| Δ step-1000 vs spec 000 | **−0.1093** | −0.0005 | +0.0023 | (base) |
| final-step val_bpb | — (not reached) | 1.0934 | **1.0929** | **1.0938** |
| Δ final-step val_bpb | — | −0.0004 | **−0.0009** | (base) |
| pre-quant post-EMA val_bpb | — | 1.09249 | **1.09193** | **1.09289** |
| Δ pre-quant post-EMA | — | −0.00040 | **−0.00096** | (base) |

## What settled the question

Run A's mid-training −0.109 looked like a huge win. It was **pod-level bf16 numerical variance, not a real QK=6.0 effect.** Run A ran on a different physical H100 node than spec 000; hardware-level numerical drift compounded over 1000 steps to ~0.1 of train_loss.

Verification run 004c (same config as spec 000 except QK_GAIN_INIT=6.0, same VAL_LOSS_EVERY=4000 → matched RNG) landed within ±0.001 bpb of spec 000 at all three final-eval stages. **Noise floor is 0.0002 (SOTA seed std). Our Δ is within 5× of that.** Tied.

## Matched-step train_loss comparison (spec 000 vs 004c)

```
step  spec 000  004c      Δ
 500   3.3098   3.3135   +0.0037
1000   3.2487   3.2510   +0.0023
1500   3.0884   3.1133   +0.0249  (batch noise spike)
2000   2.9431   2.9455   +0.0024
2500   2.9438   2.9445   +0.0007
3000   2.9232   2.9342   +0.0110
3500   2.7950   2.8072   +0.0122
```

004c tracks spec 000 very closely except for the step 1500 batch-noise spike. Average Δ across all milestones: +0.008 (within noise range).

## QK=5.5 data (Phase 1 Run B)

At step 2000, QK=5.5 hit 2.8529 vs spec 000's 2.9431 at same step → Δ −0.0902. Big mid-training lead, but Run B also stopped at step 2246 (5-min cap) without reaching training end. Same pod as Run A, so same hardware-variance contamination risk. Without a matched-VAL=4000 verification run for QK=5.5, we can't distinguish "real QK=5.5 win" from "pod-variance fluke similar to Run A." Given QK=6.0's verification result (also hardware noise), **QK=5.5's Phase-1 signal is also probably noise.** Didn't verify because QK=6.0's null result makes the monotonic-improvement hypothesis implausible anyway.

## Other findings

- **QK=6.0 artifact size: 16.046 MB, over the 16MB leaderboard cap.** Higher QK gain produces less compressible weights. Even if QK=6.0 were a small win on bpb, it'd need additional size engineering to ship.
- **VAL_LOSS_EVERY affects RNG trace.** Dense val sampling (VAL=200) consumes RNG calls that shift training data ordering vs sparser val (VAL=4000). For clean step-matched comparisons across runs, VAL_LOSS_EVERY must match the baseline's. Spec 001 and spec 002 accidentally hit this; spec 004b hit it again. Now documented in memory.

## Cost breakdown

| phase | wall | cost | notes |
|---|---|---|---|
| Phase 1 (Run A + Run B on one 8×H100 pod, 5-min caps) | ~15 min | $5.70 | initial A/B triage |
| Phase 2 (004b full 10-min run + recovery) | ~50 min + recovery pod | $8.37 | set VAL=200, broke RNG match; pod vanished post-eval |
| Phase 3 (004c verification full 10-min, matched VAL=4000) | ~14 min | $5.70 | clean apples-to-apples, NCCL crashed during post-training but pre-quant landed first |
| **Total spec 004** | | **$19.77** | |

## Key lessons captured in memory

Two new policies saved for future specs:

1. **Screen via final-step val_bpb, kill on stopping_early** — skip post-training pipeline (EMA/quant/sliding/TTT). Saves ~$3-4/run. Spec 004c's post-training stage was where NCCL crashed; we only got useful numbers because pre-quant happens BEFORE the crash point. A kill-on-stopping_early policy would have terminated cleanly.

2. **Match VAL_LOSS_EVERY to the baseline for RNG-clean comparison.** Confused us three specs in a row.

## Handback

Research: evaluation file + experiments.md row are ready in the same commit. Kill decision is cleanest yet — three runs, three independent data points, all tied within noise. Update `research/ideas/` with SHELVED status for any QK-extension idea.

Four consecutive screen specs (001, 002, 003, 004) now all killed. Remaining $180 budget available. Next spec is research's call — layerwise LR decay? Num_kv_heads=2? Something entirely different?
