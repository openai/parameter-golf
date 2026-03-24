# Asynchronous Prefetching — submission notes

## Key changes

Same model, optimizer, data layout, and training math as baseline. This is **a general purpose rework that could apply to most other approaches** for slight speed boosts. Overlap CPU data prep and host→device copies with GPU work so the GPU spends less time idle.

| Area | Original (`train_gpt_og_linux.py`) | Improved |
|------|-------------------------------------|----------|
| **Training batches** | Each step: read tokens on CPU, then H2D — all on the main thread before forward. | **Background thread** (`PrefetchingDistributedTokenLoader`) builds the **next** pinned CPU batch while the GPU runs the current step. Primary win: **CPU work overlaps GPU compute** (not GPU-side double-buffering of H2D vs forward). |
| **H2D** | Single default stream. | Optional **dedicated CUDA copy stream** (`TRAIN_COPY_STREAM`, off when timing diagnostics are on). Transfers use pinned memory; the training path **still waits for that step’s H2D** before forward (`wait_stream`). |
| **Validation** | Simple loop: slice → GPU → forward; BPB byte math on GPU. | **Prefetch thread** for pinned CPU batches; **double-buffered** H2D with copy stream + events so the **next** batch can copy while the **current** forward runs. Default **`VAL_BYTECOUNT_DEVICE=cpu`** moves BPB byte counting off the GPU vs the original (set **`cuda`** to mirror baseline GPU LUT math). |

## Diagnostics

To measure how much time this actually saves, I added **`TRAINING_TIMING_BREAKDOWN`** (batch CPU vs H2D vs FWD/BWD/opt vs val; adds syncs). When enabled, lines log every **`TRAINING_TIMING_EVERY`** steps (default 200) and for early steps (first 10). Extra logs: train/val I/O mode, `val_stage_time_ms`, train vs val wall time split.

**`VAL_BYTECOUNT_DEVICE`** defaults to **`cpu`** in the improved script (not an extra flag you must set). Use **`cuda`** if you want validation byte math on the GPU like the original.

Optional **`VAL_PROGRESS_LOG_EVERY`** (default **0**): set to a positive value to log per-batch validation progress (`val_progress:...`).

## Defaults & toggles

Overlap features are **on by default** (`TRAIN_PREFETCH`, `TRAIN_COPY_STREAM`, `VAL_PREFETCH`, `VAL_COPY_STREAM`, etc.) and can be turned off via env vars if needed. **`TRAINING_TIMING_BREAKDOWN`** defaults to 0 and is not displayed. Prefetch/overlap are **automatically disabled** when `TRAINING_TIMING_BREAKDOWN=1` so timings stay interpretable.

## Idea

**Prefetch training and validation batches asynchronously and parallelize CPU ↔ GPU transfers with compute** to minimize pipeline bubbles under a fixed wall-clock budget.
This is an intuitive idea that I came up with that could help models with real research and architectural advancements place slightly higher.

## Why this may be unimpactful in some cases

With **`TRAINING_TIMING_BREAKDOWN=1`**, early-step lines look like this (same hardware / config as above; `grad_accum_steps=8`, per-micro averages for batch/forward/backward):

```text
timing_breakdown step:1 micro_steps:8 batch_cpu_ms:0.29 batch_h2d_ms:0.35 forward_ms:30.54 backward_ms:64.93 grad_clip_ms:0.00 optimizer_ms:55.37 val_ms:121092.09 explicit_sync_ms:0.16 (per_optimizer_step; forward/backward/batch averaged over micro_steps; grad_accum_steps=8)
timing_breakdown step:2 micro_steps:8 batch_cpu_ms:0.29 batch_h2d_ms:0.35 forward_ms:30.29 backward_ms:64.72 grad_clip_ms:0.00 optimizer_ms:55.08 val_ms:0.00 explicit_sync_ms:0.00 (per_optimizer_step; forward/backward/batch averaged over micro_steps; grad_accum_steps=8)
timing_breakdown step:3 micro_steps:8 batch_cpu_ms:0.28 batch_h2d_ms:0.37 forward_ms:30.66 backward_ms:65.18 grad_clip_ms:0.00 optimizer_ms:54.45 val_ms:0.00 explicit_sync_ms:0.00 (per_optimizer_step; forward/backward/batch averaged over micro_steps; grad_accum_steps=8)
timing_breakdown step:4 micro_steps:8 batch_cpu_ms:0.31 batch_h2d_ms:0.34 forward_ms:30.34 backward_ms:64.43 grad_clip_ms:0.00 optimizer_ms:55.19 val_ms:0.00 explicit_sync_ms:0.00 (per_optimizer_step; forward/backward/batch averaged over micro_steps; grad_accum_steps=8)
```

**How to read this:** `batch_cpu_ms` and `batch_h2d_ms` are ~0.3 ms per micro-step; `forward_ms` and `backward_ms` are ~30 ms and ~65 ms per micro-step. Scaled by 8 micro-steps, batch prep + H2D is on the order of **~5 ms per optimizer step**, while forward + backward + optimizer is on the order of **~800+ ms**. So **data movement is a tiny slice** of the step; overlapping it cannot move wall-clock much when the GPU is already busy with compute for almost the whole step.

**Caveat:** On a **much faster GPU** (or smaller model / larger batch so steps are shorter), the same CPU+H2D work could become a **larger fraction** of the step, and prefetch or val overlap might show up more in profiles. The breakdown above is **not** universal; it only shows why the optimization can be a no-op when **compute is the bottleneck**.
