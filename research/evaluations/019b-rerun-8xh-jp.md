# Evaluation 019b-rerun — Literal-α 8×H100 JP seed 42 rerun

**Spec:** none — same config as spec 019b (`e93d77d`), rerun on same-day pod for seed-variance calibration
**Run:** `runs/019b-rerun-8xh-jp/seed_42/`
**Date:** 2026-04-22
**Commit:** `e93d77d`
**Status:** full pipeline completed (train + GPTQ + TTT)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 8×H100 SXM AP-JP-1 ($23.92/hr, ~35 min including TTT) |
| Final step | 4872 / 20000 (wallclock cap 596s) |
| Loop activation | step 2155 @ frac 0.350 |
| val_bpb @ step 4000 | 1.1132 |
| val_bpb @ step 4872 (in-training) | 1.0702 |
| Pre-quant post-EMA val_bpb | **1.06970** |
| Post-GPTQ val_bpb | 1.07905 |
| **Post-TTT val_bpb** | **1.06675** |

---

## Comparison to original 019b

| run | hardware | steps | pre-quant EMA | post-TTT |
|---|---|---|---|---|
| #1736 reference | 8×H JP | — | — | **1.06610** |
| 019b original | 8×H JP | 4824 | 1.06951 | **1.06628** |
| **019b rerun** | **8×H JP** | **4872** | **1.06970** | **1.06675** |

- **Pre-quant EMA Δ: +0.00019** (rerun worse — within noise)
- **Post-TTT Δ: +0.00047** (rerun worse — at the edge of seed std ~0.0002–0.0005)
- Rerun got 48 more steps (faster pod pre-loop) but landed worse post-TTT

---

## Step-matched loss vs 019b original

| step | 019b orig | 019b rerun | Δ |
|---|---|---|---|
| 2200 (post-loop) | 2.5341 | 2.5417 | +0.0076 |
| 2500 | 2.5551 | 2.5591 | +0.0040 |
| 3000 | 2.5613 | 2.5712 | +0.0099 |
| 3500 | 2.5617 | 2.5731 | +0.0114 |
| 4000 | 2.4023 | 2.4149 | +0.0126 |
| 4500 | 2.2632 | 2.2836 | +0.0204 |

Rerun runs +0.004–0.020 higher train_loss post-loop throughout. The pod drew a slightly different node — same commit, same seed, different per-step loss trajectory. This is the pod lottery in action.

---

## Throughput

| step | tok/s rerun | tok/s orig |
|---|---|---|
| pre-loop (~1000) | 8.12M | 8.11M |
| 2200 (just post-loop) | 8.04M | 7.98M |
| 3100 | 7.10M | 6.84M |
| 4000 | 6.67M | 6.68M |

Rerun was ~0.25M tok/s faster pre-loop and mid-run, converging by step 4000. No Type B stalls — 019b on this pod is clean. Yet the rerun still lands worse post-TTT despite more steps, confirming the per-step loss quality (not step count) drives the final number.

---

## Seed-variance calibration

This rerun establishes the seed/pod variance range for 019b on 8×H JP:

| seed | steps | pre-quant EMA | post-TTT |
|---|---|---|---|
| 42 (original) | 4824 | 1.06951 | **1.06628** |
| 42 (rerun, same-day) | 4872 | 1.06970 | 1.06675 |

**Observed spread: 0.00047 post-TTT.** This is ~2× the claimed SOTA std of 0.0002. The original 019b's miss of #1736 by 0.00018 is well within this spread — seed 43/44 on a good pod draw could close it.

---

## Decision

**019b is still our best result.** The rerun confirms the 0.00018 miss to #1736 is within seed variance. Seeds 43/44 are the highest-value next move.

Note: this run's `final_model.int6.ptz` (16MB) is on disk and submittable, but post-TTT 1.06675 does not beat #1736 (1.06610).
