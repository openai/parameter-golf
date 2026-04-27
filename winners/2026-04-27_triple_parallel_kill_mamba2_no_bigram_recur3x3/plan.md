# Experiment 0051_triple_parallel

Parent: 0046_kill_mamba2_middle_parallel (current best 2-seed mean 2.01031)

## Question
**Topology generalization**: 0046 placed PARALLEL only at middle (pos 1). Test if parallel-everywhere works: all 3 unique blocks are PARALLEL (ATTN || kill-Mamba-2). Compare to 0046/0050 mean 2.0103.

This is essentially Hymba-strict topology (parallel everywhere) but with kill-Mamba-2 instead of S4D-Lin and no-BG. Earlier 0025 Hymba-strict (full Mamba + parallel everywhere) lost vs sandwich. But that was full-Mamba-2 + BG; the kill+no-BG combination may behave differently.

## Hypothesis [CONJECTURE]
val_bpb in [2.005, 2.030]. Single-seed.

- val ∈ [2.005, 2.013] (compound win, parallel everywhere wins): 25% likely. New best.
- val ∈ [2.013, 2.020] (matches 0046): 30% likely. Topology saturates at middle-parallel.
- val ∈ [2.020, 2.030] (slight regression, parallel-everywhere over-expands): 35% likely.
- val > 2.030 (significant regression): 10% likely. Parallel-everywhere dilutes attention.

Cap math: parallel block ~2.44 MB int8. 3 × 2.44 = 7.32 MB SSM+attn (vs 0046's 5.74 MB). Predicted artifact ~15.8 MB. Cap-tight (16 MB).

## Change
**env.sh ONLY**:
- `PARALLEL_LAYER_POSITIONS=0,1,2` (was 1)
- `MAMBA2_LAYER_POSITIONS=` (was 0,2)

## Disconfirming
- val < 2.013: parallel everywhere compounds.
- val > 2.020: middle-parallel placement is special.
- val ∈ [2.013, 2.020]: tied with 0046.

## Notes from execution
Direct env-var-only fork.
