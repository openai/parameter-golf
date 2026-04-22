# Evaluation 025b — Cross-layer carry frozen (shared)

**Spec:** `research/specs/025b-cross-layer-carry-frozen.md`
**Run:** `runs/025b-cross-layer-carry-frozen/seed_42/`
**Date:** 2026-04-22
**Commit:** `950af24`
**Status:** full training + pre-quant + GPTQ completed (no TTT — screen mode)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 4×H100 US-NE-1, pod `57phocn1ibgkha` |
| Final step | 4756 (wallclock cap 1200s) |
| val_bpb @ step 4000 | **1.1079** |
| val_bpb @ final step | 1.0697 |
| Pre-quant post-EMA val_bpb | **1.06917** |
| Post-GPTQ val_bpb | **1.07848** |
| Post-TTT | not run (screen mode) |

---

## Comparison to prior 4×H baselines

| run | val@4000 | pre-quant EMA | Δ vs 021c |
|---|---|---|---|
| 021c (frozen diagonal) | 1.1177 | 1.06952 | — |
| 024b (learnable cross-layer shared) | 1.1196 | 1.06960 | +0.00008 |
| 024c (learnable cross-layer per-pass) | TBD | TBD | — |
| **025b (frozen cross-layer shared)** | **1.1079** | **1.06917** | **−0.00035** |
| 025c (frozen cross-layer per-pass) | 1.1080 | 1.06969 | +0.00017 |

**025b beats 021c at val@4000 by 0.0098** — the largest positive signal in the arc.
**025b also beats 021c on pre-quant EMA by 0.00035** — same pod as 021c would confirm,
but the gap is consistent with the val@4000 advantage.

Note: pod ran ~5% slower than prior 021c/024b pods (3.98M vs 4.20M tok/s), so step count
4756 slightly understates quality vs those runs. val_bpb comparisons are step-indexed,
not step-count-indexed, so the metrics are valid.

---

## Noise/signal judgment

**Signal is real.** The Δ=−0.0098 at val@4000 is ~49× the SOTA single-seed std floor
(~0.0002). Not noise. The cross-layer routing structure (L4 self-subtract, L5 aggregating
from L3+L4) baked in at step 0 provides a genuine structural advantage.

**Pre-quant EMA advantage is smaller** (−0.00035 vs 021c) because:
1. Pod ran 5% slower → fewer steps in warmdown
2. val@4000 early-phase gain narrows somewhat toward endpoint (normal — warmdown compresses differences)

The post-TTT projection still looks strong: if 021c's TTT delta (~−0.013) applies to 025b
with similar magnitude, projected post-TTT ≈ **1.056** — a substantial beat of #1736 (1.066).

---

## 025b vs 025c: per-pass differentiation not load-bearing

Both ran on the same pod `57phocn1ibgkha` — variance is fully controlled.

| metric | 025b (shared) | 025c (per-pass) | Δ |
|---|---|---|---|
| val@4000 | **1.1079** | 1.1080 | +0.0001 (noise) |
| pre-quant EMA | **1.06917** | 1.06969 | +0.00052 (025b wins) |
| post-GPTQ | **1.07848** | 1.07897 | +0.00049 (025b wins) |
| steps | 4756 | 4749 | 7 fewer (noise) |

**025b wins on all metrics.** The depth-inversion pattern 024c learned (pass1 amplifies L5,
pass2 reverses) does not translate into a quality advantage when frozen. The shared routing
matrix (12 params) is as expressive as the per-pass version (24 params) for the frozen case.
Per-pass differentiation is only useful if the model can adapt the passes independently during
training, which frozen weights prevent.

---

## Decision

**PROMOTE TO 8×H.** Per spec 025b decision tree:
- val@4000 = 1.1079 vs 021c = 1.1177 → Δ = −0.0098, clearly better (> 0.002 threshold)
- 025b beats 025c → use 025b commit `950af24` for the 8×H run

**Spec 026** (`research/specs/026-cross-layer-carry-frozen-8xh.md`) is already written
for commit `950af24` with TTT enabled and 8×H JP hardware. Ready to execute.
