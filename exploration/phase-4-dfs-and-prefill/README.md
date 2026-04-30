# Phase 4: DFS, Token Injection & Prefix State Prefill

**Dates:** Mar 25, 2026
**Goal:** Explore non-standard ways to inject more context into the model: depth-first-search ordering, token injection schemes, prefix state prefilling, and hyperparameter variants.
**Outcome:** Mixed results. Prefix state prefill showed some promise but added complexity. DFS with token injection went through 5 versions (V1-V5) with incremental improvements.

## Runs

| Run | Approach | Notes |
|-----|----------|-------|
| DFS/ | Depth-first search ordering | Reorder token processing |
| DFS-TI/ | DFS + Token Injection (V1-V5) | 5 iterations of refinement |
| prefix-state-prefill/ | Prefix state initialization | Warm-start recurrent state |
| hyperparam-variants/ | Hyperparameter exploration | LR, batch size sweeps |

## What Led to Phase 5

These experiments showed that the model's bottleneck wasn't token ordering or state initialization — it was the fundamental inability of per-token attention to capture long-range patterns cheaply. This insight led to the macro-sidechannel idea: what if we gave the model a separate pathway for summarizing and retrieving long-range context?
