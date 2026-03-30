# Phase 7: R-Series Alternatives

**Dates:** Mar 27, 2026
**Goal:** Explore alternative architecture variants (r1, r2, r4, r124, r1248) to ensure trans-hier wasn't a local optimum.
**Outcome:** Breadth of exploration. The r-series variants tested different recurrence depths and combinations.

## Runs

| Run | Config | Notes |
|-----|--------|-------|
| r1 | Single recurrence | Minimal depth |
| r2 | Double recurrence | Moderate depth |
| r4 | Quad recurrence | Deeper |
| r124 | Mixed r1+r2+r4 | Heterogeneous depth |
| r1248 | Mixed r1+r2+r4+r8 | Full spectrum |
| 8gpu-r124 | r124 on 8xH100 | Full-scale test |
| 8gpu-r1248 | r1248 on 8xH100 | Full-scale test |

## Key Findings

The r-series explored whether mixing different recurrence depths (shallow + deep) in the same model outperformed uniform depth. Results fed back into the Behemoth model iterations.
