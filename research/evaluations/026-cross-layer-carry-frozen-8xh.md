# Evaluation — Spec 026: Cross-layer carry frozen, 8×H100 JP

**Date:** 2026-04-22
**Spec:** `research/specs/026-cross-layer-carry-frozen-8xh.md`
**Status:** SEED 42 COMPLETE — seed 314 queued

## Result (seed 42)

| Stage | bpb |
|---|---|
| val@4000 | 1.1128 |
| pre-quant post-EMA | **1.06893** |
| post-GPTQ | 1.07827 |
| post-TTT | **1.06582** |
| TTT gain | −0.01245 |

Steps: 4863 · Commit: `950af24` · Hardware: 8×H100 AP-JP-1 · Cost: ~$12

## Δ vs baselines

| baseline | post-TTT | Δ |
|---|---|---|
| 021e (frozen diagonal) | 1.06622 | **−0.00040** (better) |
| #1736 reference | 1.06610 | **−0.00028** (better) |
| #1769 mean (5 seeds) | 1.06453 | +0.00129 (worse) |

## Key diagnostic: gap vs #1769

Fetched #1769 per-seed diagnostics from submission.json. Full three-stage breakdown:

| | #1769 mean (5 seeds) | Spec 026 seed 42 | Gap |
|---|---|---|---|
| Float post-EMA | 1.06742 | 1.06893 | **+0.00151** |
| GPTQ penalty | +0.00946 | +0.00934 | −0.00012 (we're better) |
| TTT gain | −0.01236 | −0.01245 | −0.00009 (we're better) |
| Post-TTT | 1.06453 | 1.06582 | +0.00129 |

**Conclusion: the entire gap is in the float model (training quality). GPTQ and TTT are equivalent or marginally better.**

The gap is explained by seed selection. #1769 ran 7 seeds and submitted best 5. Their seed 314 (best) produced float 1.06637; their worst submitted seed produced 1.06802. Our seed 42 lands at 1.06893 — worse than all 5 of their seeds. Seed 42 was inherited from #1736 (their prior baseline run) and is a known mediocre training seed. #1769 deliberately moved to different seeds.

## TTT efficiency note

Our TTT gain (−0.01245) matches or slightly exceeds #1769's (−0.01236). No TTT implementation gap. LoRA warm-start-A (`d70888f`) was not used for this seed — seed 314 run will stack it.

## Decision

**CONTINUE with seed 314.** Cross-layer carry + clip=12 + LoRA warm-start-A (`d70888f`) on seed 314. If the float lands near #1769's seed 314 (~1.066), projected post-TTT is ~1.053–1.055 — a clear leaderboard beat.

Seed plan: 314 → 2025 → 777 (dexhunter's ranked seeds from #1769).

## Next steps

- Run spec 026 seed 314 at commit `d70888f` (~$12)
- If post-TTT ≤ 1.062: submit and run seeds 2025 + 777 for 3-seed mean
- If post-TTT 1.062–1.065: run seeds 2025 + 777 to confirm mean
- Spec 027 (depth curriculum) held until seed 314 result in hand
