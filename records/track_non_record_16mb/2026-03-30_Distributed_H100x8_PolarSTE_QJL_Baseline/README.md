# Non-record Submission: Distributed 8xH100 Polar STE + QJL KV-cache Baseline

This folder captures the first end-to-end distributed Hopper baseline for the Polar STE + QJL KV-cache stack. The goal here is not a leaderboard claim. The goal is to prove that the architecture compiles, trains, exports, reloads, and runs final autoregressive KV evaluation on `8xH100 80GB HBM3` under DDP without deadlocks.

The run used:

- `WORLD_SIZE=8` via `torchrun --standalone --nproc_per_node=8`
- `QAT_SCHEME=polar`
- `WEIGHT_QUANT_SCHEME=polar`
- native `KV_QUANT_BACKEND=qjl`
- `ENABLE_TORCH_COMPILE=0`
- a hard `600s` wallclock with an internal `15s` finalization reserve

## Result

Single-seed Hopper run (`SEED=314`):

| Metric | Value |
|--------|------:|
| Steps completed | `3382` |
| Teacher-forced final `val_bpb` | `1.4594` |
| Final autoregressive `qjl` `val_bpb` | `2.12830032` |
| Final autoregressive throughput | `93.51 tok/s` |
| Artifact bytes (`polar+zlib`) | `14,751,006` |
| Total submission bytes | `14,875,186` |
| Peak VRAM allocated | `1933 MiB` |
| Peak VRAM reserved | `2080 MiB` |
| Total wallclock | `592.209s` |

The large gap between teacher-forced evaluation (`1.4594`) and final autoregressive KV evaluation (`2.1283`) is the main reason this is submitted as a non-record baseline rather than a record attempt. Training remains stable, but the quantized KV path still injects too much error during free-running decode.

## Engineering Notes

This run found and fixed a real infrastructure bug before submission:

- The first 8xH100 attempt exceeded the wallclock at `601.863s`.
- Root cause: the internal training budget reserved time for export/final-eval, but did not subtract pre-training setup overhead.
- The fix now measures `pre_training_overhead` before entering the main loop and reduces the usable training budget accordingly.
- The successful run logged `pre_training_overhead:6610ms`, `train_budget_after_setup:578390ms`, and finished at `total_wallclock:592209ms`.

This folder therefore serves as both:

- a distributed Hopper baseline for Polar STE + QJL
- a regression test for the DDP-safe final KV evaluation path and wallclock budgeting logic

## Files

- `train_gpt.py`: self-contained training + export + rank-0 KV evaluation script
- `triton_kv_ops.py`: Triton kernels kept alongside the script, even though the Hopper-winning eval backend here is native `qjl`
- `run_h100x8.sh`: exact launcher used for the successful run
- `train_seed314_budgetfix.log`: raw run log from the successful 8xH100 execution

## Run Command

From this folder on the official RunPod Parameter Golf image:

```bash
bash run_h100x8.sh
```

The launcher bakes in the validated Hopper settings:

- `KV_QUANT_BACKEND=qjl`
- `ENABLE_TORCH_COMPILE=0`
- `WARMUP_STEPS=0`
- `LR_WARMUP_STEPS=128`
- `LR_WARMUP_INIT_SCALE=0.1`
- `MAX_WALLCLOCK_SECONDS=600`
- `FINALIZE_BUDGET_SECONDS=15`

## Why This Matters

Even though the quality is not leaderboard-competitive, this submission proves that:

- Polar STE weight training survives the transition from local `1x3090` experimentation to `8xH100` DDP.
- The final autoregressive KV evaluator can run rank-0-only under distributed training without deadlock.
- Hopper prefers native `qjl` over the current Triton decode path on this `batch=1` autoregressive workload.
- The export path stays under the `16MB` artifact limit on real `8xH100` runs.
