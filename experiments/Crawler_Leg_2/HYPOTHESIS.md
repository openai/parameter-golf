# Crawler_Leg_2 Hypothesis

Date: 2026-03-30
Builds on: Crawler_Leg_1 (11-arm arch sweep) + Crawler_Ablations_v1 (post-training policies)

## Mission

Test whether the two dominant Leg 1 wins (mlp=5.0 −0.098, loops=3 −0.088) are additive when
combined, and whether the Ablations_v1 GPTQ stack (LOOP_AWARE_GPTQ −0.040, COMPILE −0.026)
transfers to the new architecture. Also probe one step further in each direction (loops=2, mlp=6.0).

## Expected outcome if additive

Leg 1 deltas (both measured independently against same baseline 1.74636):
- mlp=5.0 alone:  −0.098 → 1.64868
- loops=3 alone:  −0.088 → 1.65890

If additive: CL2-01 (loops=3 + mlp=5.0) should land near 1.74636 − 0.186 ≈ **1.560 BPB**
If subadditive (likely — both reduce quant gap, there may be diminishing returns):
  likely in the 1.58–1.62 range

## Theory

Both mlp=5.0 and loops=3 independently reduce the quantization gap (from 0.354 baseline):
- mlp=5.0: gap 0.287 (kernel tile efficiency → faster steps → better-trained weights)
- loops=3: gap 0.288 (less weight-sharing pressure → cleaner weight distributions)

The mechanisms are partially orthogonal (width vs sharing pressure), so some additivity
is expected. But since both reduce the quant gap via different paths to the same floor,
there will be diminishing returns — the gap cannot go below ~0.20 regardless of architecture.

## Arms

| ID | Config | SKIP_GPTQ | Hypothesis |
|----|--------|:---------:|------------|
| CL2-00 | baseline (loops=4, mlp=4.0) | 1 | Fresh pod reference, should match CL1-00 ≈ 1.746 |
| CL2-01 | loops=3 + mlp=5.0 | 1 | PRIMARY: combined wins, expect ~1.56–1.62 |
| CL2-02 | loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ + COMPILE | 0 | Full stack: add −0.040 + −0.026 from Ablations_v1 |
| CL2-03 | loops=2 + mlp=5.0 | 1 | Push loops: another −0.085 quant gap reduction? |
| CL2-04 | loops=3 + mlp=6.0 | 1 | Push MLP: diminishing returns vs 5.0, or more? |

## Hard Rules

1. `DELTA_NET_HEADS=0` — DeltaNet quarantined.
2. `NGRAM_EVAL_ORDER=0` — ngram eval off. Legal only.
3. `SKIP_EMA=1` — EMA is confirmed harmful (+0.070 BPB from Ablations_v1).
4. `CRAWLER_QUANT_INT8=1` — mandatory QAT (disabling costs +0.197 BPB from Leg 1).
5. All arms: 600s wallclock, seed 1337, 1 GPU.
6. Key metric: `final_int6_sliding_window_exact` (lower = better).

## Exit Criteria

- CL2-01 delta vs CL2-00 > −0.100: wins are at least partially additive → proceed to CL2-02
- CL2-01 delta vs CL2-00 < −0.050: subadditive, understand why before proceeding
- CL2-02 delta vs CL2-01: expect −0.050 to −0.070 (LOOP_AWARE_GPTQ + COMPILE combined)
- CL2-03 wins vs CL2-01: loops=2 is viable → add to production config
- CL2-04 wins vs CL2-01: mlp=6.0 viable → check submission size stays under 16MB

## Next after Leg 2

If CL2-01 + CL2-02 both win → Crawler_Leg_3:
  - 3-seed confirmation of best config
  - Rascal III submission candidate
