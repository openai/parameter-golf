## 03 — Training System

Code references:

- `train_gpt.py`
- `train_gpt_windows.py`
- `data_utils.py`
- `eval_utils.py`
- `optimizer_utils.py`
- `quant_utils.py`

## Runtime entry

On Windows, training is launched via `train_gpt_windows.py`, which:

- disables unsafe flash SDP path for this environment,
- patches distributed backend behavior for Windows safety,
- loads patched modules before running `train_gpt.py`.

## Data pipeline

`data_utils.py` provides:

- shard loading with header checks,
- rank-aware shard partitioning for distributed mode,
- reproducible/randomized data seeding behavior,
- `DistributedTokenLoader.next_batch(...)` with pinned CPU -> GPU transfer.

## Optimizer routing

`train_gpt.py` splits parameters into optimizer families:

1. **Muon** for dense matrix parameters (core feature learning pressure)
2. **AdamW** token embedding / LM head group(s)
3. **AdamW** scalar/control/LoRA groups

This routing is intentional and architecture-aware.

## Stabilization and scheduling

Major training controls include:

- wallclock-aware cosine schedule,
- warmup/maturity ramp,
- recurrence-aware gradient scaling (`grad / active_steps`),
- grad clipping,
- optional dynamic LR scaling using grad norm,
- optional adaptive loss filter for outlier micro-batches,
- EMA shadow weights used for evaluation/export.

## Evaluation and export

`eval_utils.py` supports:

- sliding-window evaluation,
- optional TTT adaptation diagnostics,
- BPB computation with byte-aware accounting.

`train_gpt.py` export path supports:

- best-checkpoint policy by validation BPB,
- raw `.pt` save,
- int8 quantized + zlib compressed `.ptz`,
- optional FP vs int8(dequantized) quality comparison.
