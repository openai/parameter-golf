## Summary

This PR adds a new non-record submission candidate: `2026-04-03_DeepFloor`.

DeepFloor is a compact recurrent multi-view language model that replaces a large flat stack with:

- repeated QKV+O recurrent blocks
- two cross-token modes: periodic floor attention or a fused recurrent accumulator
- explicit stability controls: contraction target, decay, TBPTT chunking, Jacobian penalty, periodic normalization, and optional HiPPO-style fused state cores
- a frozen record-folder submission snapshot so the submission entrypoint runs without importing mutable repo-root code

## Why this is interesting

This is not a leaderboard-quality run yet. The value of the submission is that it packages a real, unusual architecture direction that is reproducible inside the challenge record format:

- depth recurrence / universal-transformer-style computation
- multi-view recurrence instead of standard layer stacking
- a self-contained submission path built for cold pods and remote validation

## Included artifacts

- `README.md`
- `RESULTS.md`
- `submission.json`
- `train_gpt.py`
- `deepfloor_snapshot.py`
- `freeze_submission_snapshot.py`
- runner scripts for local, smallbox, and fullbox validation
- train log and candidate result json

## Current checked-in evidence

Best real-`enwik8` small-box matrix candidate:

- `fused_d32_v2`
- `val_bpb = 7.9221`
- `artifact_bytes = 8448`

Frozen submission preflight:

- `bytes_total = 56221`
- `bytes_model_estimated = 8192`
- `bytes_code = 48029`

Checked-in candidate submission:

- `submission.json` built from `candidate_result_seed1337.json`
- `bytes_total = 56477`
- `val_bpb = 7.9221`

## Validation

- local submission preflight on the frozen record-folder snapshot
- remote GPU submission preflight on a reused small pod
- remote small-box suite on real `enwik8`
- remote `8x H100` fullbox suite, with synced `smoke`, `matrix`, `launch_logs`, and `evolution` artifacts under `runs/fullbox/`

## Fullbox evidence

Best fullbox recipe-search result:

- `frontier seed 2025`
- `val_bpb = 4.1101`
- `test_bpb = 4.0239`
- `artifact_mb = 0.1377`
- mode: `floor`
- state core: `scalar_decay`
- recurrent dim: `96`
- views: `8`

This is stronger evidence that DeepFloor is a real direction worth reviewing, but it is still not presented as a record claim or as three repeated fixed-candidate contest runs.

## Submission lane

This should be reviewed as a `track_non_record_16mb` research submission, not a record claim.
