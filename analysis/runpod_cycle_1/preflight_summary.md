# Preflight Summary (V5.9 canonical strict run)

- Command:
  - `python3 records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py --mode preflight`
- Timestamp (UTC from payload): `2026-03-23T21:57:21.901895+00:00`
- Result: **FAIL** (`hard_failures=1`)

## Captured Logs
- Stdout: `analysis/runpod_cycle_1/raw/preflight_stdout.log`
- Stderr: `analysis/runpod_cycle_1/raw/preflight_stderr.log`

## Required Environment Summary
- CUDA visibility:
  - `cuda_available=false`
  - `device_count=0`
  - `minimum_cuda_devices=1`
  - Check outcome: `fail` (`cuda.runtime`)
- Disk space:
  - Path: `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/fullrun_outputs`
  - Free: `41.756 GB`
  - Minimum required: `40.0 GB`
  - Check outcome: `pass`
- Dataset/tokenizer paths:
  - `data_path`: `data/datasets/fineweb10B_sp1024` (exists)
  - `localbench_data`: `data/datasets/fineweb10B_sp1024_localbench_v1` (exists)
  - `tokenizer_path`: `data/tokenizers/fineweb_1024_bpe.model` (exists)
- Writable output dirs:
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/fullrun_outputs` (pass)
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/fullrun_outputs/T0` (pass)
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/fullrun_outputs/T1` (pass)
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/fullrun_outputs/D` (pass)

## Stop Condition
- Strict preflight failed on a hard requirement (`cuda.runtime`), so paid-cycle execution was halted before T0, per packet rules.
