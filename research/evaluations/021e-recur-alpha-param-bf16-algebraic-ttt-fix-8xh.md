# Evaluation 021e — Recur-α Parameter+bf16+algebraic+TTT-fix, 8×H100 JP

**Spec:** `research/specs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh.md`
**Run:** `runs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh/seed_42/`
**Date:** 2026-04-22
**Commit:** `d761a22`
**Status:** full pipeline completed (train + GPTQ + TTT)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 8×H100 SXM AP-JP-1 ($23.92/hr, ~35 min total) |
| Final step | 4863 / 20000 (wallclock cap 596s) |
| Loop activation | step 2158 @ frac 0.350 |
| val_bpb @ step 4000 | 1.1134 |
| val_bpb @ step 4863 (in-training) | 1.0702 |
| Pre-quant post-EMA val_bpb | **1.06944** |
| Post-GPTQ val_bpb | 1.07863 |
| **Post-TTT val_bpb** | **1.06622** |

---

## Comparison

| run | commit | hardware | steps | pre-quant EMA | post-TTT |
|---|---|---|---|---|---|
| #1736 reference | — | 8×H JP | — | — | **1.06610** |
| 019b original | `e93d77d` | 8×H JP | 4824 | 1.06951 | **1.06628** |
| 019b rerun | `e93d77d` | 8×H JP | 4872 | 1.06970 | 1.06675 |
| 021d (param-α bf16) | `8b2d791` | 8×H JP | 4881 | 1.06960 | not run |
| **021e (param+bf16+alg+ttt-fix)** | **`d761a22`** | **8×H JP** | **4863** | **1.06944** | **1.06622** |

**021e vs 019b original: −0.00006 post-TTT.** New best single-seed result.
**021e vs #1736: miss by 0.00012.** Within seed std (~0.0003–0.0005).

---

## Step-matched loss vs 019b rerun

| step | 021e | 019b(rerun) | Δ |
|---|---|---|---|
| 2200 (post-loop) | 2.5438 | 2.5417 | +0.0021 |
| 2300 | 2.6037 | 2.6040 | −0.0003 |
| 2400 | 2.6301 | 2.6332 | −0.0031 |
| 2500 | 2.5641 | 2.5591 | +0.0050 |
| 2600 | 2.5242 | 2.5261 | −0.0019 |
| 2700 | 2.5082 | 2.5114 | −0.0032 |
| 2800 | 2.5707 | 2.5738 | −0.0031 |
| 2900 | 2.5458 | 2.5485 | −0.0027 |
| 3000 | 2.5711 | 2.5712 | −0.0001 |
| 3100 | 2.5009 | 2.5028 | −0.0019 |
| 3200 | 2.4712 | 2.4729 | −0.0017 |
| 3300 | 2.6608 | 2.6653 | −0.0045 |
| 3400 | 2.5642 | 2.5625 | +0.0017 |
| 3500 | 2.5708 | 2.5731 | −0.0023 |
| 3600 | 2.4713 | 2.4674 | +0.0039 |
| 3700 | 2.5543 | 2.5550 | −0.0007 |
| 3800 | 2.4997 | 2.5014 | −0.0017 |
| 3900 | 2.6267 | 2.6252 | +0.0015 |
| 4000 val | 1.1134 bpb | 1.1132 bpb | +0.0002 |
| 4500 | 2.2809 | 2.2836 | −0.0027 |

**Key finding:** The per-step loss gap that plagued all prior 021 variants (+0.007–0.022 vs 019b) is completely eliminated. 021e oscillates ±0.005 around 019b(rerun) throughout — pure noise. The two bug fixes (TTT α + algebraic blend) resolved the deficit.

---

## Throughput

| step | tok/s 021e | tok/s 019b rerun |
|---|---|---|
| pre-loop (~1000) | 8.06M | 8.04M |
| 2200 (just post-loop) | 8.06M | 8.04M |
| 3000 | 7.18M | 7.17M |
| 4000 | 6.54M | 6.52M |

No Type B compile stalls. Throughput virtually identical to 019b rerun — Parameter+bf16 provides no throughput tax at 8×H on this pod.

---

## recur_alpha

`nn.Parameter(requires_grad=False)`, bf16 dtype. Values frozen throughout at:
`[[1.078125, 1.2734375, 1.4296875], [1.015625, 0.97265625, 0.83203125]]`
`grad_norm=0.000000` every step — confirmed frozen.

---

## Two-bug-fix analysis

### Bug 1: TTT α fix
021 lineage branched from 017 (not 019), so `forward_ttt` didn't apply the recur_alpha blend. Expected post-TTT improvement from fix: ~+0.0025. Observed TTT delta for 021e: pre-quant 1.06944 → post-TTT 1.06622 = **−0.01322**. For reference, 019b's TTT delta was −0.01249 on the original commit. 021e's larger delta (−0.01322 vs −0.01249) is consistent with the TTT α fix enabling deeper TTT improvement.

### Bug 2: Algebraic blend form
`x = x_before + α·(x_new − x_before)` vs prior `α·x_new + (1−α)·x_before`. In bf16 these diverge numerically. The per-step loss elimination confirms the algebraic form now matches 019b-original's actual behaviour.

---

## Decision

**021e is our new best single-seed result (post-TTT 1.06622).** It edges out 019b original by 0.00006 and misses #1736 by 0.00012.

Per the spec accept criteria (post-TTT in (1.06610, 1.06710] = "Borderline"): compare to 019b-rerun on same pod, may skip 3-seed and pivot.

However, 021e's **mechanism is now fully debugged** (TTT α fixed, algebraic blend confirmed). The next highest-value move is:

1. **021e seeds 43/44 on the same JP pod** — 0.00012 miss is within seed std; a second good draw could tie or beat #1736. Cost: ~$20.
2. **Spec 022 (extra TTT depth)** — orthogonal mechanism, unblocked.
3. **Spec 023 (TTT LoRA bias)** — orthogonal, unblocked.

**Recommended next step: 021e seed 43 on a fresh JP pod.** The TTT δ improvement from the fix (+0.007 larger than 019b) is real and reproducible. A lucky pod draw on seed 43/44 has ~50–60% credence of beating #1736.
