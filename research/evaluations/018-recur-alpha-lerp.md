# Evaluation — Spec 018 (Recur-Alpha torch.lerp optimization)

**Run dir:** `runs/018-recur-alpha-lerp/run-c-lerp/`
**Commit:** `97d9854` on `exp/recur-alpha-lerp`
**Pod:** `g5s1rqfhia58uk` — 2×H100 SXM, US-NE-1, NA volume `hvpdph5i3g`
**Eval date:** 2026-04-21

## Hypothesis recap

Replacing `x = alpha * x_new + (1.0 - alpha) * x_before` with `x = torch.lerp(x_before, x_new, alpha)` should fuse the 4-op blend into a single CUDA primitive, reducing memory traffic and kernel launch overhead.

## Result

| Run | Commit | Config | Avg last 5 tok/s | vs baseline | vs current blend |
|-----|--------|--------|-----------------|-------------|-----------------|
| A (016b) | 154c9b8 | no recur-alpha | 3,333K | — | — |
| B (016b) | 4dd2d63 | current 4-op blend | 3,234K | −2.9% | — |
| **C (018)** | **97d9854** | **torch.lerp** | **3,252K** | **−2.4%** | **+0.6% recovered** |

Steps at stabilization: ~step 100. Readings at 375/400/425/450/475: 3,256K / 3,250K / 3,251K / 3,252K / 3,254K.

## Decision criterion outcome

Blend overhead on proxy model (B vs A): ~99K tok/s = 3.0% of baseline.
lerp recovered: ~18K tok/s = **18% of blend overhead recovered**.

Per spec's decision table:
- ≥40% recovered: big win ✗
- 20-40% recovered: partial win ✗
- **<20% recovered: disappointing ← actual (18%)** ✓

## Interpretation

torch.lerp is **marginally better** than the manual 4-op blend but not by much. Two likely explanations:

1. **torch.compile was already partially fusing** the manual blend with surrounding ops. The remaining unfused portion is small; lerp squeezed it but not dramatically.
2. **Memory bandwidth is not the bottleneck** at 6L/256d — the proxy model's matmuls are so cheap that the blend op's memory traffic is already a small absolute cost. The ~0.6% gain may be amplified at full model scale where matmuls dominate even more (making blend overhead fraction smaller, not larger).

On the full 11L/512d model, lerp likely recovers <0.3% of total throughput — enough to apply, but not a game-changer.

## Decision — APPLY lerp in full pipeline

**Ship lerp.** It's a 2-line code change, mathematically identical, and never hurts. The <20% overhead recovery is "disappointing" per spec criteria but the cost of applying it is zero. Any future full-pipeline spec (017-style) should use commit `97d9854` or later as the base.

The bake-in refactor (018b) is worse — see `research/evaluations/018b-recur-alpha-bakein.md`.

## Cost

| item | cost |
|---|---|
| Run C: ~6 min compile + 2 min training | ~$0.80 |
| **018 total** | **~$0.80** |

## Cross-references

- Spec: `research/specs/018-recur-alpha-lerp.md`
- Control data (Runs A, B): `research/evaluations/016b-recur-alpha-throughput.md`
- Follow-up: `research/evaluations/018b-recur-alpha-bakein.md`
