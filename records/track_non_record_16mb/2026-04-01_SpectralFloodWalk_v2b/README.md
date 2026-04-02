# Spectral Flood Walk v2b

This record folder packages the first `V2b` persistent hidden-memory branch of Spectral Flood Walk in the same RunPod-friendly shape as the earlier exploratory folders.

The purpose of `V2b` is narrower than "try more memory":

- keep a strong transformer host
- let persistent hidden-space memory grow online across the validation stream
- spend extra eval FLOPs on post-score maintenance of touched memory slots
- measure whether that extra runtime work actually improves `val_bpb`

This is still a **non-record exploratory** package. The point is to answer whether the coprocessor-style framing has real signal before we spend time on a submission branch.

## Target Hardware For This Version

This folder is versioned for **1×H100 exploratory runs**.

The wrappers default to:

- `SFW_TARGET_HARDWARE=1xH100 exploratory`
- `SFW_TARGET_GPU_COUNT=1`
- `SFW_NPROC_PER_NODE=1`

That keeps the discovery workflow cheap and fast. If we later want an `8xH100` validation branch, that should be a deliberate override or a separate folder.

## What This Run Is

`train_gpt.py` in this folder is a thin wrapper around the repo-root [spectral_flood_walk_v2b.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v2b.py).

The root runner contains:

- the same strong host transformer used in the `v2a` family
- persistent hidden-space memory keyed by hashed multi-order context
- score-first analytic hidden-gradient updates after scoring
- delayed read gating via `memory_min_read_count`
- optional maintenance passes that refine touched memory slots
- honest `val_bpb`
- eval-time FLOP estimates for:
  - memory lookup
  - memory updates
  - memory maintenance

## Fast Workflow

These are the intended first pod experiments:

```bash
./runpod_preflight.sh
./runpod_smoke.sh
./runpod_baseline.sh
./runpod_gate.sh
./runpod_flop_push.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

That sequence is intentional:

1. make sure the pod is healthy
2. confirm the wrapper stack works end to end
3. establish the raw hidden-memory line
4. test whether read-gating helps
5. push maintenance FLOPs harder once the mechanism is warm

To compare the core staged profiles in one shot:

```bash
./runpod_compare_core.sh
python3 ../../../tools/summarize_v2b_runs.py runs/*
```

For repeated seeds on one profile:

```bash
./runpod_three_seeds.sh
```

By default that uses `runpod_flop_push.sh`. To run three gate-only seeds instead:

```bash
SFW_PROFILE_SCRIPT=runpod_gate.sh ./runpod_three_seeds.sh
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

- `gate2_maint2_slots128`

Meaning:

- read gate of `2`
- two maintenance passes
- refine up to `128` touched slots per order

This is the first intentionally compute-heavier profile.

### `runpod_hits.sh`

Profile name:

- `gate2_maint2_slots128_hits`

Meaning:

- same maintenance depth as `runpod_flop_push.sh`
- touched slots prioritized by reads instead of writes

This tests whether the read path is identifying the best places to spend extra maintenance compute.

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
