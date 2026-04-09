# Spectral Flood Walk v2b

This record folder packages the first `V2b` persistent hidden-memory branch of Spectral Flood Walk in the same RunPod-friendly shape as the earlier exploratory folders.

A short retrospective of what held up and what did not is in [RESULTS.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-01_SpectralFloodWalk_v2b/RESULTS.md).

The purpose of `V2b` is narrower than "try more memory":

- keep a strong transformer host
- let persistent hidden-space memory grow online across the validation stream
- spend extra eval FLOPs on post-score maintenance of touched memory slots
- measure whether that extra runtime work actually improves `val_bpb`

This is still a **non-record exploratory** package. The point is to answer whether the coprocessor-style framing has real signal before we spend time on a submission branch.

## Target Hardware For This Version

This folder now defaults to **auto-detecting the pod**.

The common wrapper resolves:

- `SFW_TARGET_GPU_COUNT=auto`
- `SFW_TARGET_HARDWARE=auto`
- `SFW_NPROC_PER_NODE=auto`

So a single launched run will use `torchrun` when the pod exposes multiple GPUs, unless you pin it back down:

```bash
SFW_TARGET_GPU_COUNT=1 SFW_NPROC_PER_NODE=1 ./runpod_gate4.sh
```

For `v2b`, the more time-efficient option on large pods is usually **multiple independent queued runs**, because the online memory-growth eval path is mostly a rank-0 sequential workload.

## What This Run Is

`train_gpt.py` in this folder is a thin wrapper around the repo-root [spectral_flood_walk_v2b.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v2b.py).

The root runner contains:

- the same strong host transformer used in the `v2a` family
- persistent hidden-space memory keyed by hashed multi-order context
- score-first analytic hidden-gradient updates after scoring
- delayed read gating via `memory_min_read_count`
- optional maintenance passes that refine touched memory slots with loss-prioritized replay
- honest `val_bpb`
- eval-time FLOP estimates for:
  - memory lookup
  - memory updates
  - memory maintenance

## Fast Workflow

These are the intended first pod experiments:

```bash
./runpod_preflight.sh
./runpod_queue_parallel_core.sh
```

That queued path is intentional:

1. make sure the pod is healthy
2. fill separate GPUs with `baseline`, `gate2`, and `gate4`
3. summarize without babysitting the pod

To compare the core staged profiles in one shot:

```bash
./runpod_compare_core.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

For repeated seeds on one profile:

```bash
./runpod_three_seeds.sh
```

By default that uses `runpod_gate4.sh`. To run a different profile instead:

```bash
SFW_PROFILE_SCRIPT=runpod_gate4.sh ./runpod_three_seeds.sh
```

For parallel gate-only seeds on separate GPUs:

```bash
./runpod_queue_parallel_gate4_seeds.sh
```

## Matrix Tooling

The wrapper scripts cover the core staged profiles, but the broader curated matrix lives in the repo tools.

Show the staged matrix:

```bash
python3 ../../../tools/generate_v2b_matrix.py
```

Generate shell commands for the whole matrix:

```bash
python3 ../../../tools/generate_v2b_matrix.py --format shell --python-bin python3 --output-dir runs
```

Generate a wider search from the checked-in tweakable search-space file:

```bash
python3 ../../../tools/generate_v2b_matrix.py \
  --matrix-json ../../../tools/v2b_search_space.json \
  --format shell \
  --python-bin python3 \
  --output-dir runs
```

That path exists so candidate numbers live in data instead of source edits. If a knob feels like a variable, it should usually move into the search-space file.

The current replay-maintenance defaults intentionally favor sharp slot-local updates:

- `maintenance_use_grad=false`
- `maintenance_replay_depth=2`
- `maintenance_grad_mix=0.25` only matters when `maintenance_use_grad=true`

So the default pod profiles now run pure replay sharpening unless a sweep explicitly asks for EMA help.

Summarize completed runs:

```bash
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

The matrix note is in [2026-04-01-spectral-flood-walk-v2b-matrix.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/plans/2026-04-01-spectral-flood-walk-v2b-matrix.md).

## Core Profiles

### `runpod_baseline.sh`

Profile name:

- `baseline_memread1_nomaint`

Meaning:

- immediate reads
- no maintenance passes

This is the raw persistent-memory reference line.

### `runpod_gate.sh`

Profile name:

- `gate2_nomaint`

Meaning:

- require one prior scored write before reading
- no maintenance passes

This tests whether a small warm-up period reduces the early-chunk tax.

### `runpod_gate4.sh`

Profile name:

- `gate4_nomaint`

Meaning:

- stronger read gate
- no maintenance passes

This is the more conservative warm-up ablation.

### `runpod_flop_push.sh`

Profile name:

- `gate2_replay_loss_maint2_slots128_nograd`

Meaning:

- read gate of `2`
- two replay-sharpen passes
- prioritize hard cases by loss
- pure replay maintenance by default
- refine up to `128` touched slots per order

This is the first intentionally compute-heavier profile.

### `runpod_flop_push_gate4.sh`

Profile name:

- `gate4_replay_loss_maint2_slots128_nograd`

Meaning:

- the same maintenance budget as `runpod_flop_push.sh`
- but on top of the stronger `gate4` warm-up
- still using pure replay updates unless overridden

This is the main gate-4 replay-sharpen profile for the next pod.

### `runpod_hits.sh`

Profile name:

- `gate2_replay_hits_maint2_slots128_nograd`

Meaning:

- same maintenance depth as `runpod_flop_push.sh`
- touched slots prioritized by reads instead of hard-case loss

This tests whether the read path is identifying the best places to spend extra maintenance compute.

### `runpod_hits_gate4.sh`

Profile name:

- `gate4_replay_hits_maint2_slots128_nograd`

Meaning:

- `gate4` warm-up
- `hits`-prioritized replay sharpening

This is the explicit follow-up if `gate4` remains the best gate-only branch.

## Queue Scripts

These scripts are meant for the "pod startup is the expensive part" case.

### `runpod_queue_parallel_core.sh`

Uses separate GPUs for:

- `runpod_baseline.sh`
- `runpod_gate.sh`
- `runpod_gate4.sh`

Default GPU assignment:

- `SFW_QUEUE_GPUS="0 1 2"`

### `runpod_queue_parallel_gate4_seeds.sh`

Uses separate GPUs for parallel `gate4` seeds.

Defaults:

- `SFW_QUEUE_GPUS="0 1 2"`
- `SFW_QUEUE_SEEDS="1337 42 2025"`

## What The Result Means

The key fields in `result.json` are:

- `eval_context.val_bpb`
- `eval_online_persistent_hidden.val_bpb`
- `eval_delta_online_bpb`
- `eval_online_persistent_hidden.memory_total_flops_estimate`
- `eval_online_persistent_hidden.memory_maintenance_flops_estimate`
- `eval_online_persistent_hidden.readable_slots_mean`
- `eval_online_persistent_hidden.persistent_memory.readable_fraction`

Decision rule:

- if gated memory beats immediate reads, the early-chunk tax is real
- if maintenance-heavy runs improve `delta_online`, extra eval FLOPs are doing useful work
- if maintenance FLOPs rise sharply but `delta_online` stays flat, we are lighting up the hardware without growing a better model yet

## Promotion

Once you decide which run is worth freezing into the record folder:

```bash
./promote_run.sh runs/<timestamp>_<profile>_seed<seed>
```

## References

- [spectral_flood_walk_v2b.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v2b.py)
- [generate_v2b_matrix.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/tools/generate_v2b_matrix.py)
- [summarize_v2b_runs.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/tools/summarize_v2b_runs.py)
