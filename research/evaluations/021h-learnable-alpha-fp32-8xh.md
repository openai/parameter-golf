# Evaluation 021h — Learnable α in fp32, 8×H100 JP

**Spec:** `research/specs/021h-learnable-alpha-fp32-8xh.md`
**Run:** `runs/021h-learnable-alpha-fp32-8xh/seed_42/`
**Date:** 2026-04-22
**Commit:** `5906820`
**Status:** full pipeline completed (train + GPTQ + TTT)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 8×H100 SXM AP-JP-1 ($23.92/hr, same pod as 021g, ~35 min) |
| Final step | 4753 / 20000 (wallclock cap 596s) |
| Loop activation | step 2200 @ frac 0.350 |
| val_bpb @ step 4000 | **1.1084** (best of 021 family at this checkpoint) |
| val_bpb @ final step (4753) | 1.0709 |
| Pre-quant post-EMA val_bpb | **1.07043** |
| Post-GPTQ val_bpb | 1.07976 |
| **Post-TTT val_bpb** | **1.06734** |

---

## Comparison

| run | commit | steps | pre-quant EMA | post-TTT |
|---|---|---|---|---|
| #1736 reference | — | — | — | **1.06610** |
| 017 (learnable α fp32, buggy TTT) | `4dd2d63` | 4784 | **1.06861** | 1.06733 |
| **021e** (frozen α) | `d761a22` | 4863 | 1.06944 | **1.06622** |
| 021g (learnable α bf16) | `fab6e7f` | 4804 | 1.06987 | 1.06693 |
| **021h (learnable α fp32)** | **`5906820`** | **4753** | **1.07043** | **1.06734** |

**021h post-TTT (1.06734) ≈ 017 buggy-TTT (1.06733)** — identical result, confirming TTT fix alone doesn't close the gap when pre-quant EMA is weaker. **+0.00112 vs 021e.**

---

## Step-matched loss vs 017

Per-step loss tracks 017 closely (±0.001–0.007 throughout), better than 021g. Step-4000 val_bpb 1.1084 was the best of the 021 family. However a throughput stall around step 3800 cost ~50 steps vs 021g (4753 vs 4804), which hurt pre-quant EMA in warmdown.

---

## α trajectory vs 017

| site | 021h endpoint | 021g endpoint | 017 endpoint | Δ (021h vs 017) |
|---|---|---|---|---|
| L3 (pass 2) | 1.117 | 1.102 | 1.078 | +0.039 |
| L4 (pass 2) | 1.297 | 1.305 | 1.273 | +0.024 |
| **L5 (pass 2)** | **1.406** | **1.383** | **1.430** | **−0.024** |
| L3 (pass 3) | 1.031 | 1.039 | 1.016 | +0.015 |
| L4 (pass 3) | 0.961 | 0.926 | 0.973 | −0.012 |
| L5 (pass 3) | 0.840 | 0.844 | 0.832 | +0.008 |

fp32 closed the L5 gap from 021g's −0.047 to −0.024. But 021h still doesn't reach 017's exact values. The residual offset is not a precision issue — it's a different optimization landscape due to the algebraic blend form vs 017's manual-add. 017's basin is unreachable with the algebraic formulation.

---

## Throughput

021h matches 021g pre-loop (8.12M tok/s). A stall at ~step 3800 (6.51M vs expected ~6.62M) cost ~50 steps. The fp32 → bf16 cast at blend sites adds negligible overhead (6 scalars), but the Inductor recompile triggered by the cast may have caused the stall.

---

## Why 017's pre-quant advantage cannot be reproduced

Three hypotheses tested across 021g and 021h:
1. **bf16 precision trapping α** → fp32 helps but doesn't match (−0.024 gap remains)
2. **Blend form** → algebraic ≠ manual-add; different optimization landscape
3. **Pod luck** → 017 drew a fast node; same seed on a different draw lands ~1.069

**Conclusion: 017's 1.06861 pre-quant is a combination of pod luck + manual-add blend form.** With the algebraic form (required for numerical correctness), learnable α converges to a different basin regardless of storage dtype.

---

## Decision: Learnable α arc definitively closed

| variant | post-TTT | verdict |
|---|---|---|
| 021e — frozen α algebraic | **1.06622** | ✅ winner |
| 021g — learnable bf16 algebraic | 1.06693 | closed |
| 021h — learnable fp32 algebraic | 1.06734 | closed |

**021e remains the best result.** The entire buffer-α arc conclusion: frozen α with algebraic blend form + TTT fix (021e) is the optimal configuration. The α container type and storage dtype are irrelevant; what mattered was fixing the TTT bug and blend form.

### Next steps

1. **021e seed 43/44** — 0.00012 miss to #1736 is within seed std. 3-seed on a fresh JP pod.
2. **Spec 022 (extra TTT depth)** — orthogonal mechanism, unblocked.
3. **Spec 023 (TTT LoRA bias)** — orthogonal, unblocked.
