# H100 Entropy Half-Run Follow-Up — 2026-03-26

## Objective
Test whether entropy-based cache mixing can outperform the current legal PPM baseline under the H100 `half_run` budget.

## Context
- Started from the stabilized H100 cloud path with working FlashAttention and the no-compile frontier trainer path already in place.
- Existing half-run references before this sweep:

| Run | Setting | `legal_ttt val_bpb` |
| --- | --- | --- |
| `h100_ppm_entropy_fixed_*` | fixed entropy gating | `2.506169` |
| `h100_ppm_entropy_order_*` | order-adaptive entropy gating | `2.506917` |
| `h100_ppm_half_run_reconfirm` | plain multiorder PPM | `2.655119` |

## New Runs

| Run | Key settings | `legal_ttt val_bpb` | Read |
| --- | --- | --- | --- |
| `h100_ppm_entropy_fixed_center330` | `CAUSAL_CACHE_MIXING=entropy`, `CAUSAL_CACHE_ENTROPY_CENTER=3.30` | `2.50644063` | Lowering entropy center did not help; slight regression versus the incumbent fixed-entropy best. |
| `h100_ppm_entropy_fixed_steeper` | `CAUSAL_CACHE_MIXING=entropy`, `CAUSAL_CACHE_ENTROPY_SLOPE=2.60`, `CAUSAL_CACHE_ALPHA_MIN=0.08`, `CAUSAL_CACHE_ALPHA_MAX=0.42` | `2.50622178` | Very close to the best prior fixed-entropy run, but not a new best. |

## Current Best Legal Half-Run Baseline
- `sota_plus_ppm_entropy_fixed`
- `legal_ttt val_bpb: 2.506169`

## Takeaways
- Entropy gating is clearly better than plain multiorder.
- Fixed entropy remains the best current half-run branch.
- Order-adaptive entropy did not beat fixed entropy in current tests.
- Micro-tuning entropy-only knobs is producing diminishing returns.
- Next gains likely require changing the adaptation schedule, evaluation protocol, or promotion strategy rather than continuing small entropy sweeps.

## Recommended Next Direction
- Freeze entropy as the default half-run promotion branch.
- Stop spending cycles on tiny entropy-center / slope sweeps unless bundled with a larger idea.
- Focus next on promotion to stronger runs and on bigger structural levers.
