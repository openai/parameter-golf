# Final Handoff

This directory is the cleaned handoff for the current best candidate.

## Decision

Decision `A`: the current candidate remains the best thing to back next.

Why it still stands:

- The integrated evaluator was audited and two real correctness bugs were fixed before finalization:
  - the sliding-window schedule now matches the research evaluator
  - multi-rank document sharding is now non-overlapping
- The integrated path now has local verification for single-rank parity, multi-rank aggregation, and the world-size-1 TTT loop used in the single-H100 probes.
- The strongest existing proxy evidence still points in the same direction: on a strong 1xH100 checkpoint, the BOS-reset adaptive cache improved post-TTT bpb on three distant 1M-token validation slices.
- The last focused replacement check did not beat it. A smoothed bigram plus unigram-backoff variant was worse on the local proxy checkpoint and was not promoted.

## Primary Candidate

### `dominationv2_cache`

Upstream `DominationV2` training and export stack, plus one evaluation-time change:

- final validation runs in sliding-window mode with a BOS-reset online bigram cache
- cache strength is capped by `alpha`, grows with observed bigram counts via `total / (total + tau)`, and is scaled by normalized model entropy
- the cache is applied after the quantized roundtrip and after optional TTT
- the cache does not change training weights or the exported artifact

Default settings in the packaged launcher:

- `CACHE_ENABLED=1`
- `CACHE_ALPHA=0.20`
- `CACHE_TAU=8`
- `CACHE_ENTROPY_POWER=1.0`
- `TTT_ENABLED=1`
- `TTT_EPOCHS=3`
- `TTT_LR=1e-4`

## What Is Demonstrated

- Local 4080 verifier passes:
  - exact parity between integrated and research evaluators for baseline sliding eval
  - exact parity for cache eval
  - exact multi-rank aggregation back to the single-rank answer for `world_size in {2,3,8,32}`
  - exact TTT-loop equivalence in the `world_size=1` subset setting used in the 1xH100 probes
- Prior 1xH100 proxy evidence after TTT still favors this cache:
  - `start_token=0`: `1.61700285 -> 1.61190207`
  - `start_token=20000000`: `1.58458543 -> 1.57923265`
  - `start_token=40000000`: `1.56495388 -> 1.56022034`
- On the middle slice, `alpha=0.20` remained the best value in the tested sweep:
  - `0.10`: `1.58013994`
  - `0.20`: `1.57923535`
  - `0.30`: `1.57976811`
  - `0.40`: `1.58128413`

## What Is Still Inferred

- Full end-to-end score on the official full validation run under the 8xH100 budget is not yet demonstrated.
- Multi-rank evaluation aggregation is verified, but multi-rank TTT behavior has not been separately parity-tested against the research script.
- The added code bytes are materially larger than the upstream control, so artifact margin must be checked on the real run.
- Final runtime and score transfer from the 1xH100 proxy to the official 8xH100 setting still need confirmation.

## Contents

- `dominationv2_cache/train_gpt.py`
  - packaged candidate training, export, quant-roundtrip, optional TTT, and final evaluation
- `dominationv2_cache/train_8xH100.sh`
  - intended launcher for the next 8xH100 run
- `dominationv2_cache/eval_checkpoint.py`
  - checkpoint probing wrapper around the research evaluator
- `dominationv2_cache/README.md`
  - operator runbook with exact commands, expected outputs, and caveats

Supporting verification code lives outside this directory:

- `research/eval_doc_cache.py`
- `research/verify_domv2_cache.py`

## Next Step

Use [`dominationv2_cache/README.md`](dominationv2_cache/README.md) as the runbook for the next 8xH100 execution. If that run lands with a clean artifact margin, expected cache/TTT log markers, and a competitive `final_roundtrip_exact val_bpb`, then this package should be converted into the actual submission folder under `records/`.
