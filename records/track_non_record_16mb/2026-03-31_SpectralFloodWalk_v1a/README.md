# Spectral Flood Walk v1a

This record folder packages the first `V1a` semantic-memory branch of Spectral Flood Walk in the same RunPod-friendly shape as the `v0` experiments.

The purpose of `V1a` is narrow and explicit:

- replace the `v0` recurrent controller with a small transformer
- test static product-key semantic memory before adding episodic growth
- keep the pod workflow fast enough that we can compare baseline vs semantic memory cleanly

This is still a **non-record** track package. It is meant to answer whether semantic memory helps enough to justify the next stage, not to claim a leaderboard result yet.

## Target Hardware For This Version

This folder is intentionally versioned for **1×H100 exploratory runs**, not for the final `8xH100` submission machine.

That means the wrappers default to:

- `SFW_TARGET_HARDWARE=1xH100 exploratory`
- `SFW_TARGET_GPU_COUNT=1`
- `SFW_NPROC_PER_NODE=1`

and they record both the target and detected GPU counts in `notes.txt`. If the pod does not match the versioned target, the wrapper prints a warning before training starts.

When we are ready for an `8xH100` validation branch, that should be a deliberate override or a separate record folder, not an accidental reuse of the discovery wrappers.

## What this run is

`train_gpt.py` in this folder is a thin wrapper around the repo-root [spectral_flood_walk_v1.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v1.py). The root script contains the actual model and training loop:

- causal transformer controller
- optional product-key semantic memory inside selected layers
- honest `val_bpb` computation
- artifact export via `model_int8.npz`
- size worksheet in the run output
- CUDA telemetry when available

There is no seed bank or episodic pool in `V1a`. This stage is semantic-memory only.

## RunPod Preflight

Before spending real credits, run the lightweight hardware screen in
[docs/runpod-preflight.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/runpod-preflight.md).

This folder also includes the wrapper:

```bash
./runpod_preflight.sh
```

It writes the raw output to `runs/<timestamp>_preflight.log`.

## Fast Workflow

These are the intended first pod experiments:

```bash
# fast DDP / artifact smoke test
./runpod_smoke.sh

# transformer baseline with semantic memory fully off
./runpod_baseline.sh

# one semantic layer
./runpod_semantic1.sh

# two semantic layers (current "full" V1a profile)
./runpod_full.sh

# run baseline -> semantic1 -> full in one shot
./runpod_compare_triplet.sh
```

For repeated seeds on the same profile:

```bash
./runpod_three_seeds.sh
```

By default that uses `runpod_full.sh`. To run three baseline seeds instead:

```bash
SFW_PROFILE_SCRIPT=runpod_baseline.sh ./runpod_three_seeds.sh
```

Each wrapper creates a fresh run directory under `runs/` and writes:

- `train.log`
- `result.json`
- `model_int8.npz`
- `command.sh`
- `notes.txt`

To summarize the resulting `result.json` files side by side:

```bash
python3 ../../../tools/summarize_v1a_runs.py runs/*
```

Once you decide which run is worth freezing into the record folder, promote it with:

```bash
./promote_run.sh runs/<timestamp>_<profile>_seed<seed>
```

## Step 4 Commands

From a RunPod shell in `/workspace/parameter-golf`:

```bash
cd records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v1a
./runpod_baseline.sh
./runpod_semantic1.sh
./runpod_full.sh
# or just:
./runpod_compare_triplet.sh
```

That sequence is intentional:

1. baseline transformer floor
2. one semantic layer
3. two semantic layers

We want paired evidence before spending on wider sweeps.

If you intentionally want to exercise the 8-GPU launcher later, do it explicitly:

```bash
SFW_TARGET_HARDWARE='8xH100 validation' \
SFW_TARGET_GPU_COUNT=8 \
SFW_NPROC_PER_NODE=8 \
./runpod_full.sh
```

That keeps the discovery-vs-validation distinction visible in both the logs and the notes file.

## Sizing Worksheet

Before a real run, you can sanity-check the design math with:

```bash
python3 ../../../tools/spectral_flood_walk_v1_sizing.py \
  --vocab-size 1024 \
  --embed-dim 256 \
  --num-layers 6 \
  --semantic-layers 2,4 \
  --pk-num-subkeys 64 \
  --pk-key-dim 16 \
  --pk-code-dim 64 \
  --json
```

That prints:

- compact model byte estimate
- expanded semantic-memory byte estimate
- semantic layer count
- product-key entry count per layer

## Current Defaults

The shell wrappers currently target a modest but serious `V1a` profile:

- `embed_dim=256`
- `num_layers=6`
- `num_heads=8`
- `ff_mult=4`
- `seq_len=128`
- `semantic_layers=2,4`
- `pk_num_subkeys=64`
- `pk_key_dim=16`
- `pk_code_dim=64`

These are meant to be a starting point, not sacred numbers.

## Notes

- `result.json` includes the training history, throughput, evaluation metrics, and the `size_estimate` block from the worksheet logic.
- The `model_int8.npz` artifact is exported automatically so we can track approximate artifact pressure during these sweeps.
- `V1a` does not yet test episodic memory, append-only eval writes, or GPU routing.

## Expected Next Step

Once we have real pod logs for:

1. baseline
2. one semantic layer
3. two semantic layers

the next decision is simple:

- if semantic memory does not help, stop here and pivot
- if it helps clearly, move on to `V1b` same-GPU episodic memory
