# Spectral Flood Walk v1b

This record folder packages the first `V1b` episodic-memory branch of Spectral Flood Walk.

The purpose of `V1b` is narrow and explicit:

- keep the `V1a`-style transformer controller
- add same-GPU append-only episodic memory at evaluation time
- compare three eval modes from one trained controller:
  - `controller`
  - `raw`
  - `refined`

This is still a **non-record** exploratory package. It is meant to answer whether dumb writes plus fixed-function refinement help enough to justify a later learned coprocessor.

## Target Hardware For This Version

This folder is intentionally versioned for **1×H100 exploratory runs**.

The wrappers default to:

- `SFW_TARGET_HARDWARE=1xH100 exploratory`
- `SFW_TARGET_GPU_COUNT=1`
- `SFW_NPROC_PER_NODE=1`

They record both the target and detected GPU counts in `notes.txt`.

## What This Run Is

`train_gpt.py` in this folder is a thin wrapper around the repo-root [spectral_flood_walk_v1b.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v1b.py).

The root script contains:

- a small transformer controller
- optional reuse of `V1a` semantic-memory layers, default off
- append-only same-GPU episodic memory during eval
- a fixed-function bucket refiner that creates summary entries
- honest `val_bpb`
- artifact export via `model_int8.npz`
- CUDA telemetry and episodic-memory telemetry

## Evaluation Modes

Every full `V1b` run evaluates three modes on the same trained controller:

- `controller`
  - no episodic memory
- `raw`
  - append-only raw entries, no summaries
- `refined`
  - append-only raw entries plus fixed-function summary refinement

That means one run answers the main `V1b` question directly:

> does refinement help more than raw append, and do either help more than the controller alone?

## RunPod Preflight

Before spending real credits, run:

```bash
./runpod_preflight.sh
```

It writes the raw output to `runs/<timestamp>_preflight.log`.

## Fast Workflow

The intended first pod experiments are:

```bash
# fast mechanical smoke
./runpod_smoke.sh

# default exploratory V1b run
./runpod_full.sh

# longer higher-signal run
./runpod_long.sh
```

For repeated seeds on the same profile:

```bash
./runpod_three_seeds.sh
```

By default that uses `runpod_full.sh`. To run three long seeds instead:

```bash
SFW_PROFILE_SCRIPT=runpod_long.sh ./runpod_three_seeds.sh
```

Each wrapper creates a fresh run directory under `runs/` and writes:

- `train.log`
- `result.json`
- `model_int8.npz`
- `command.sh`
- `notes.txt`

To summarize the resulting `result.json` files:

```bash
python3 ../../../tools/summarize_v1b_runs.py runs/*
```

To freeze a run into the record folder:

```bash
./promote_run.sh runs/<timestamp>_<profile>_seed<seed>
```

## Step 4 Commands

From a RunPod shell in `/workspace/parameter-golf`:

```bash
cd records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v1b
./runpod_preflight.sh
./runpod_smoke.sh
./runpod_full.sh
```

If the full run is promising, follow with:

```bash
./runpod_long.sh
python3 ../../../tools/summarize_v1b_runs.py runs/*
```

## Current Defaults

The shell wrappers currently target this `V1b` controller profile:

- `embed_dim=256`
- `num_layers=6`
- `num_heads=8`
- `ff_mult=4`
- `seq_len=128`
- `use_semantic_memory=false`

And this episodic-memory profile:

- `eval_modes=controller,raw,refined`
- `episodic_bucket_count=512`
- `episodic_max_entries=65536`
- `episodic_topk=16`
- `episodic_read_alpha=0.20`
- `maintenance_every=16`
- `maintenance_budget_buckets=16`
- `maintenance_source_limit=64`
- `summary_per_bucket=4`
- `merge_similarity=0.94`

These are meant to be a starting point, not sacred numbers.

## What To Look For

The key fields in `result.json` live under `eval_modes`:

- `eval_modes.controller.val_bpb`
- `eval_modes.raw.val_bpb`
- `eval_modes.raw.delta_vs_controller_bpb`
- `eval_modes.refined.val_bpb`
- `eval_modes.refined.delta_vs_controller_bpb`

The next block to inspect is `eval_modes.refined.memory`:

- raw entry count
- summary entry count
- raw / summary memory MB estimates
- average raw candidates per query
- average summary candidates per query
- maintenance time

## Expected Next Decision

After a small set of `V1b` runs, the decision should be straightforward:

- if `raw` and `refined` both lose to `controller`, stop and pivot again
- if `raw` helps but `refined` does not, the write object may be good but the refiner is bad
- if `refined` beats `raw` and `controller`, move toward a learned local coprocessor
