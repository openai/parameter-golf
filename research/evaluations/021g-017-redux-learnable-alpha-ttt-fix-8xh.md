# Evaluation 021g — 017-redux: learnable α + TTT fix + algebraic form, 8×H100 JP

**Spec:** `research/specs/021g-017-redux-learnable-alpha-ttt-fix-8xh.md`
**Run:** `runs/021g-017-redux-learnable-alpha-ttt-fix-8xh/seed_42/`
**Date:** 2026-04-22
**Commit:** `fab6e7f`
**Status:** full pipeline completed (train + GPTQ + TTT)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 8×H100 SXM AP-JP-1 ($23.92/hr, ~35 min total) |
| Final step | 4804 / 20000 (wallclock cap 596s) |
| Loop activation | step 2200 @ frac 0.350 |
| val_bpb @ step 4000 | 1.1134 (from in-training log) |
| Pre-quant post-EMA val_bpb | **1.06987** |
| Post-GPTQ val_bpb | 1.07928 |
| **Post-TTT val_bpb** | **1.06693** |

---

## Comparison

| run | commit | steps | pre-quant EMA | post-TTT |
|---|---|---|---|---|
| #1736 reference | — | — | — | **1.06610** |
| 017 (learnable α, buggy TTT) | `4dd2d63` | 4784 | **1.06861** | 1.06733 |
| 019b original | `e93d77d` | 4824 | 1.06951 | **1.06628** |
| 021e (frozen α, algebraic, TTT fix) | `d761a22` | 4863 | 1.06944 | **1.06622** |
| **021g (learnable α bf16, algebraic, TTT fix)** | **`fab6e7f`** | **4804** | **1.06987** | **1.06693** |

**021g vs 021e: +0.00071 post-TTT.** Learnable bf16 α is worse than frozen α.
**021g vs 017: +0.00040 pre-quant EMA.** 017's pre-quant advantage did NOT reproduce.

---

## Step-matched loss vs 017

| step | 021g | 017 | Δ |
|---|---|---|---|
| 100 | 3.6469 | 3.6329 | +0.014 |
| 600 | 2.6791 | 2.6797 | −0.001 |
| 1000 | 2.8116 | 2.8125 | −0.001 |
| 1500 | 2.6489 | 2.6393 | +0.010 |
| 2000 | 2.6738 | 2.6688 | +0.005 |
| 2200 (loop on) | 2.5444 | 2.5304 | +0.014 |
| 2500 | 2.5578 | 2.5559 | +0.002 |
| 3000 | 2.5715 | 2.5644 | +0.007 |
| 3500 | 2.5662 | 2.5614 | +0.005 |
| 4000 | 2.4078 | 2.4076 | +0.000 |
| 4500 | 2.2754 | 2.2705 | +0.005 |

Pre-loop gap ±0.001–0.014 (pure noise). Post-loop gap closes to ±0.000–0.010, with **gap → 0 by step 4000**. Warmdown trajectories virtually identical.

---

## α trajectory vs 017

| step | 021g α (L0) | 017 α (L0) | Δ L5 |
|---|---|---|---|
| 2200 (first post-loop) | [0.707, 0.828, 0.672] | [0.742, 0.875, 0.813] | −0.141 |
| 2500 | [1.086, 1.320, 1.352] | [1.055, 1.289, 1.406] | −0.054 |
| 3000 | [1.102, 1.305, 1.383] | [1.078, 1.281, 1.430] | −0.047 |
| 3500 | — | [1.078, 1.273, 1.430] | — |

021g's α converges to a **similar but offset basin**: pass-2 L5 lands at ~1.383 vs 017's 1.430 (−0.047). This offset persists and doesn't close. In bf16, the LSB at α≈1.0 is 1/128 = 0.0078 — each AdamW update (grad × lr × momentum ≈ 1e-4 to 1e-5) is 10–100× smaller than one bf16 step, so updates mostly round to zero and α is trapped on a coarse grid. This is the bf16 precision hypothesis for 021h.

---

## Throughput

| step | tok/s 021g | tok/s 017 |
|---|---|---|
| 2100 (pre-loop) | 8.12M | 8.07M |
| 2200 (post-loop) | 8.04M | 7.95M |
| 3000 | 7.07M | 7.04M |
| 4000 | 6.55M | 6.53M |
| 4500 | 6.41M | 6.39M |

021g runs ~0.02–0.09M faster than 017 throughout — consistent with nn.Parameter vs literal α. No Type B stalls. Smooth monotone throughput decline.

---

## Step count: 4804 vs 017's 4784

021g got 20 more steps than 017 (expected from slightly better throughput). Not meaningful at warmdown.

---

## Key finding: 017's pre-quant advantage was pod luck

017's pre-quant EMA of 1.06861 is **0.00126 better than 021g's 1.06987**, despite 021g using the identical learnable-α mechanism. The only difference is the pod draw. 017 landed on a particularly fast node that enabled more effective α optimization within the wallclock cap. This is pod variance, not mechanism.

**021h hypothesis:** bf16 precision traps learnable α on a coarse grid (LSB 0.0078 >> AdamW step 1e-5). fp32 storage restores precision and may allow α to reach 017's basin reproducibly.

---

## Decision

**Learnable bf16 α arc closed.** 021g is worse than 021e (frozen α). The buffer/Parameter/learnable container question is now:
- Frozen α (021e): post-TTT 1.06622 ← current best
- Learnable bf16 α (021g): post-TTT 1.06693 ← worse
- Learnable fp32 α (021h): pending — the hypothesis fix

**021h launches immediately on same pod.** If 021h reproduces 017's pre-quant EMA (~1.0686), the fp32 precision hypothesis is confirmed and 021h becomes the submission candidate. If 021h also misses, 017's advantage was pod-specific and 021e 3-seed is the submission path.
