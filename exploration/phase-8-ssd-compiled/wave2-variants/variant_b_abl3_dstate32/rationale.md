# Variant B: Ablation-3 Champion + d_state=32

## Parent
ablation-3 (bifurcated A_log init) -- val_bpb=1.873 in 5 min on 1xH100

## Hypothesis
Two independent findings should stack:

1. **Bifurcated A_log init** (ablation-3's win): 25% induction heads (A_log ~ -4.0, near-infinite memory for copy/retrieval) + 75% local heads (A_log in 0.3-0.6, fast decay for local patterns). This beat log-uniform init by creating strong specialization pressure. val_bpb=1.873 vs 1.939+ for other ablations.

2. **d_state=32** (our earlier proven config): Our sweep experiments used d_state=32 and showed strong results (val_bpb=2.139 at d=1280). However, the ablation scripts inherited d_state=64 from iter-005.5's defaults. This means the ablation-3 result at 1.873 was achieved with d_state=64, not 32.

## The Confound
The ablation experiments all ran with d_state=64 (visible in ablation-1's config log line). This is different from our earlier sweeps which used d_state=32. The ablation results are therefore not directly comparable to our sweep results -- they differ in both the ablated variable AND d_state.

## What This Variant Tests
By setting d_state=32 while keeping bifurcated A_log, we isolate whether:
- d_state=64 was contributing to ablation-3's strong result (if this variant is worse)
- d_state doesn't matter much (if this variant matches)
- d_state=32 is actually better, as suggested by our sweeps (if this variant wins)

## Parameter Budget Impact
d_state=32 vs 64 reduces:
- `in_proj` output dimension: 6320 -> 6256 (saves 64 * 1536 = 98K params per in_proj)
- `theta_proj` output dimension: 32 -> 16 (saves 16 * 1536 = 25K params)
- SSM state size: halved (faster per-step compute, less memory)
- Total savings: ~123K params, freeable for wider model or more iterations

The smaller state also means faster SSD scan (state matmuls are d_state x headdim), so throughput should increase slightly, meaning more tokens seen in the same wall-clock budget.

## Decision Tree
```
IF val_bpb < 1.85: d_state=32 + bifurcated is the optimal combo. Use for submission.
ELIF val_bpb 1.85-1.90: d_state=32 matches d_state=64 -- state dimension not critical.
ELIF val_bpb > 1.90: d_state=64 was helping. Revert to ablation-3 as-is.
```

## Run Command
```bash
# 1xH100 smoke test (5 min)
RUN_ID=variant_b_abl3_dstate32 MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 full run (10 min)
RUN_ID=variant_b_abl3_dstate32_full MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Change Summary
Single change from ablation-3: `D_STATE` default 64 -> 32 (line 82 of Hyperparameters class).
All other code identical -- bifurcated A_log init, BCNorm, RoPE on B/C, trapezoidal discretization, Triton SSD kernel, QAT, Muon optimizer, etc.
