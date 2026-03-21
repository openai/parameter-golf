# Results Ledger Schema

## Purpose
This file defines the schema for the experiment ledger used by the controller and cloud worker.

Recommended ledger file:
- `results.jsonl`

Format:
- one JSON object per line
- append-only
- each line represents one completed or failed run

## Required Fields
- `run_id`
  - string
  - unique identifier for the run

- `timestamp_utc`
  - string
  - ISO 8601 timestamp for when the run result was recorded

- `status`
  - string
  - one of:
    - `success`
    - `failed`
    - `invalid_size`
    - `missing_metrics`
    - `crashed`

- `family`
  - string
  - experiment family such as:
    - `batch_size`
    - `schedule`
    - `optimizer`
    - `architecture`
    - `quantization`
    - `precision_budget`
    - `context_length`

- `hypothesis`
  - string
  - one-sentence explanation of what the run is testing

- `change_summary`
  - string
  - short human-readable summary of the code or env change

- `gpu_type`
  - string
  - example: `RTX 5090`

- `wallclock_budget_s`
  - number
  - intended training budget for the run

- `train_shards`
  - number
  - number of train shards available for that run

- `val_bpb`
  - number or null
  - parsed from `final_int8_zlib_roundtrip_exact`

- `val_loss`
  - number or null
  - parsed from `final_int8_zlib_roundtrip_exact`

- `eval_time_ms`
  - number or null
  - parsed from the final static quantized eval line when available

- `bytes_total`
  - number or null
  - parsed from `Total submission size int8+zlib`

- `peak_memory_mib`
  - number or null
  - parsed from `peak memory allocated`

- `step_reached`
  - number or null
  - training step reached when the run ended

- `judge`
  - string
  - one of:
    - `promote`
    - `archive`
    - `reject`
    - `retry`

- `judge_note`
  - string
  - short explanation of why the run was promoted, archived, rejected, or marked for retry

## Strongly Recommended Fields
- `base_commit`
  - string
  - git commit hash used as the run base

- `diff_summary`
  - string
  - concise summary of the diff relative to the previous champion

- `env`
  - object
  - only include the env vars intentionally changed for this run

- `log_path`
  - string
  - path to the raw log file used for parsing

- `worker_id`
  - string
  - pod id or machine identifier

- `parent_run_id`
  - string or null
  - the run this experiment was derived from

## Example Record
```json
{
  "run_id": "search_001",
  "timestamp_utc": "2026-03-21T05:00:00Z",
  "status": "success",
  "family": "batch_size",
  "hypothesis": "A smaller batch may improve optimization efficiency on single-GPU search hardware.",
  "change_summary": "Reduced TRAIN_BATCH_TOKENS from 524288 to 262144.",
  "gpu_type": "RTX 5090",
  "wallclock_budget_s": 300,
  "train_shards": 10,
  "val_bpb": 1.2284,
  "val_loss": 2.0813,
  "eval_time_ms": 18450,
  "bytes_total": 15800412,
  "peak_memory_mib": 4120,
  "step_reached": 1210,
  "judge": "archive",
  "judge_note": "Completed successfully but did not beat the current champion.",
  "base_commit": "48404cd",
  "diff_summary": "Batch-size-only experiment.",
  "env": {
    "TRAIN_BATCH_TOKENS": 262144,
    "MAX_WALLCLOCK_SECONDS": 300
  },
  "log_path": "/workspace/parameter-golf/logs/search_001.txt",
  "worker_id": "runpod-5090-a",
  "parent_run_id": "champion_000"
}
```

## Policy
- Every run should be written to the ledger, even failures.
- Never overwrite previous entries.
- If parsing fails, write a partial record with `status` set appropriately.
- The ledger is part of the experiment memory and should be treated as a research asset.
