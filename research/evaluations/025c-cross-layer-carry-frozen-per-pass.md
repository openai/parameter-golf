# Evaluation 025c — Cross-layer carry frozen, per-pass

**Spec:** `research/specs/025c-cross-layer-carry-frozen-per-pass.md`
**Run:** `runs/025c-cross-layer-carry-frozen-per-pass/seed_42/`
**Date:** 2026-04-22
**Commit:** `414cbc3`
**Status:** full training + pre-quant + GPTQ completed (no TTT — screen mode)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 4×H100 US-NE-1, pod `57phocn1ibgkha` (same as 025b) |
| Final step | 4749 (wallclock cap 1200s) |
| val_bpb @ step 4000 | **1.1080** |
| val_bpb @ final step | 1.0702 |
| Pre-quant post-EMA val_bpb | **1.06969** |
| Post-GPTQ val_bpb | **1.07897** |
| Post-TTT | not run (screen mode) |

---

## 025c vs 025b: per-pass differentiation null result

Ran on the **same pod as 025b** — pod variance is fully controlled.

| metric | 025b (shared) | 025c (per-pass) | Δ (025c − 025b) |
|---|---|---|---|
| val@4000 | 1.1079 | 1.1080 | +0.0001 (noise) |
| pre-quant EMA | **1.06917** | 1.06969 | **+0.00052 (025b wins)** |
| post-GPTQ | **1.07848** | 1.07897 | +0.00049 (025b wins) |
| steps | 4756 | 4749 | −7 (noise) |

025c is marginally **worse** than 025b on all non-noise metrics. 025b wins.

---

## Noise/signal judgment

The pre-quant EMA gap of +0.00052 is ~2.6× the SOTA single-seed std floor (~0.0002).
This is a small but real signal that the per-pass parameterization is not helping and
may be mildly hurting. The val@4000 tie (+0.0001) is indistinguishable from noise.

**Mechanistic read:** Freezing per-pass values encodes the depth-inversion pattern (pass1
amplifies L5, pass2 reverses) that 024c discovered. But at freeze time this specific routing
may not generalize — the model trained under shared routing might use the residual stream
differently than 024c did when learning the per-pass values. The shared routing of 025b is
more robust because it was validated to work (as a frozen structure) over the full training
arc, whereas 025c's per-pass structure is a direct lift from a different model's training
trajectory.

---

## Decision

**Per spec 025c decision tree: "Roughly equal → Prefer 025b (simpler) for 8×H."**
In fact 025c is slightly worse, confirming 025b as the correct variant.

Per-pass differentiation adds no benefit when frozen. The structural load-bearing content
is the cross-layer routing pattern (L4 self-subtract, L5 aggregation) — not the per-pass
depth inversion. This is captured fully by 025b's 12-parameter shared matrix.

**Next step: execute spec 026** (8×H JP full pipeline, commit `950af24`, TTT enabled).
025c is shelved; no 8×H promotion.
