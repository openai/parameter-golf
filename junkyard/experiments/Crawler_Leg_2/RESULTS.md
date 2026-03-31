# Crawler_Leg_2 Results

**Date:** 2026-03-30
**Hardware:** 8×H100, 350s wallclock/arm (~4400–5600 steps/arm)
**Seed:** 1337
**Baseline:** loops=4, mlp=4.0, SKIP_GPTQ=1

## Summary Table

| ID | Config | Steps | Pre-quant BPB | Int6 SW BPB | Delta vs CL2-00 |
|----|--------|------:|:-------------:|:-----------:|:---------------:|
| CL2-00 | baseline (loops=4 mlp=4.0 SKIP_GPTQ=1) | 4421 | 1.2147 | 1.20285 | — |
| CL2-01 | loops=3 + mlp=5.0 (SKIP_GPTQ=1) | 4833 | 1.2147 | 1.20211 | −0.0007 |
| CL2-02 | loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ=1 + COMPILE=1 | 4844 | 1.2142 | **1.19593** | **−0.0069** ✅ BEST |
| CL2-03 | loops=2 + mlp=5.0 (SKIP_GPTQ=1) | 5595 | 1.2197 | 1.20667 | +0.0038 ❌ |
| CL2-04 | loops=3 + mlp=6.0 (SKIP_GPTQ=1) | 4710 | 1.2115 | 1.19828 | −0.0046 |

## Key Findings

### 1. CL2-02 wins — full stack is the config (1.19593)
loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ + COMPILE_FULLGRAPH is the production config.
CL2-02 beats baseline by −0.0069. This is the smallest absolute delta we've seen at
8×H100 scale, but the config ranking is clear and consistent with CL1 direction.

### 2. Architecture wins compress at 8×H100 scale
CL2-01 (loops=3 + mlp=5.0 combined) only beats CL2-00 by −0.0007. In CL1 at 1×H100:
loops=3 alone = −0.088, mlp=5.0 alone = −0.098. At 8×H100/350s with ~5× more steps,
the model is better trained — the architecture ceiling is hit faster, making the
architectural differentials compress significantly. The GPTQ stack adds the clean −0.0062
on top (CL2-02 vs CL2-01).

### 3. LOOP_AWARE_GPTQ + COMPILE adds −0.0062 (CL2-02 vs CL2-01)
At 8×H100 scale: 1.20211 → 1.19593. Loop-aware GPTQ completed in 5.5s (fast at 8 GPUs).
Fullgraph compile confirmed safe with NGRAM removed.

### 4. loops=2 is worse (CL2-03: +0.0038)
loops=3 remains the optimal loop count. Fewer loops = less representational capacity
at this scale. Confirms CL1 finding: loops=3 is the sweet spot.

### 5. mlp=6.0 is competitive but not best (CL2-04: −0.0046)
mlp=6.0 alone (SKIP_GPTQ=1) outperforms mlp=5.0 alone (−0.0046 vs −0.0007). However
CL2-02 (mlp=5.0 + full GPTQ stack) wins overall at −0.0069. A mlp=6.0 + LOOP_AWARE_GPTQ
arm was not run — this is an open question for Leg 3 or Bandit_Wagon.

## Note on Q_GAP Column

The run_all.sh summary shows Q_GAP = +0.000 for all arms. This is a script artifact:
the summary script reports val_bpb = int6_sw_bpb (same value). The actual quant delta
(pre-quant DIAGNOSTIC BPB vs final int6 SW BPB) is negative (SW benefit dominates quant
loss) for all arms, meaning CRAWLER_QUANT_INT8=1 QAT is working as designed.

## Submission Size Context

| ID | Submission Size |
|----|----------------|
| CL2-00 | 8.90 MB |
| CL2-01 | 8.67 MB |
| CL2-02 | 9.84 MB |
| CL2-03 | 8.80 MB |
| CL2-04 | 9.05 MB |

CL2-02 at 9.84 MB is comfortably under the 16 MB limit. GPTQ increases compressed size
relative to naive int6 because GPTQ-optimized weights have less zstd redundancy.

## Production Config (locked for Bandit_Wagon / Leg 3)

```
CRAWLER_LOOPS=3
CRAWLER_MLP_MULT=5.0
COMPILE_FULLGRAPH=1
LOOP_AWARE_GPTQ=1
CRAWLER_QUANT_INT8=1
```

All settings confirmed by CL2-02 = 1.19593 int6 SW BPB.

## Open Questions for Crawler_Leg_3

1. **mlp=6.0 + LOOP_AWARE_GPTQ**: CL2-04 used SKIP_GPTQ=1. Does mlp=6.0 beat mlp=5.0
   when the full GPTQ stack is added? Expected to be competitive with or beat CL2-02.
2. **3-seed confirmation**: CL2-02 is single-seed. Need seeds 1337/42/7 for submission candidate.
3. **Width lever**: Bandit_Wagon BW-01/02 (dim=576/640) with production config — does
   wider base model further improve beyond 1.19593?

## Prior Reference (different scale — 1×H100/600s)

| System | Int6 SW BPB | Notes |
|--------|:-----------:|-------|
| CL1-00 baseline | 1.74636 | 1×H100, 817 steps |
| CL1-07 mlp=5.0 | 1.64868 | 1×H100, 917 steps |
| CL1-01 loops=3 | 1.65890 | 1×H100, 884 steps |
| **CL2-02 full stack** | **1.19593** | **8×H100, 4844 steps** |
