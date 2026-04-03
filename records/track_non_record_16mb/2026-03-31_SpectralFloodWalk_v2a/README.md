# Spectral Flood Walk v2a

This record folder packages the first `V2a` branch of Spectral Flood Walk:

- a stronger transformer spine inspired by the competition-leading 11-layer family
- explicit multi-order residual tables
- three clean eval modes in one run:
  - `context`
  - `static residual`
  - `online residual`

This is still a **non-record exploratory** package. The purpose is not to claim leaderboard quality yet. The purpose is to answer whether directly predictive residual memory helps on top of a serious transformer base.

## Target Hardware For This Version

This folder is intentionally versioned for **1×H100 exploratory runs**.

The wrappers default to:

- `SFW_TARGET_HARDWARE=1xH100 exploratory`
- `SFW_TARGET_GPU_COUNT=1`
- `SFW_NPROC_PER_NODE=1`

That distinction is important. `V2a` is currently about learning whether the residual-table mechanism has signal, not about spending 8-GPU money before we know the answer.

## What This Run Is

`train_gpt.py` in this folder is a thin wrapper around the repo-root [spectral_flood_walk_v2a.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/spectral_flood_walk_v2a.py).

The root script contains:

- a stronger transformer spine with GQA, RoPE, relu² MLPs, and U-Net-style skip reuse
- two spine variants:
  - `plain`
  - `xsa`
- a shared residual basis `U`
- hashed multi-order residual tables
- a calibration phase that fits residual memory on top of the frozen base model
- honest `val_bpb`
- artifact export for both:
  - `model_int8.npz`
  - `residual_tables.npz`

## Fast Workflow

These are the intended first pod experiments:

```bash
./runpod_preflight.sh
./runpod_smoke.sh
./runpod_spine_a.sh
./runpod_spine_b.sh
python3 ../../../tools/summarize_v2a_runs.py runs/*
```

Or, after preflight:

```bash
./runpod_compare_pair.sh
python3 ../../../tools/summarize_v2a_runs.py runs/*
```

Each training run compares all three eval modes automatically:

- `eval_context`
- `eval_static_residual`
- `eval_online_residual`

So one run answers more than one question.

## Step 4 Commands

From a RunPod shell in `/workspace/parameter-golf`:

```bash
cd records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a
./runpod_preflight.sh
./runpod_spine_a.sh
./runpod_spine_b.sh
python3 ../../../tools/summarize_v2a_runs.py runs/*
```

That sequence is intentional:

1. hardware sanity check
2. plain strong spine
3. same-family stronger flavored spine
4. compare whether residual tables help on either

## Current Defaults

The wrapper defaults are intentionally stronger than `V1a/V1b`:

- `model_dim=512`
- `num_layers=11`
- `num_heads=8`
- `num_kv_heads=4`
- `mlp_mult=3`
- `seq_len=512`
- `stride=64`
- `residual_rank=16`
- `residual_orders=1,2,3,4`
- `residual_table_size=65536`

These are still exploratory defaults, not final submission settings.

## What The Result Means

The important fields in `result.json` are:

- `eval_context.val_bpb`
- `eval_static_residual.val_bpb`
- `eval_online_residual.val_bpb`
- `eval_delta_static_bpb`
- `eval_delta_online_bpb`

Decision rule:

- if both deltas are positive on both spines, stop and pivot
- if `static` helps but `online` does not, the value object is promising but the online update rule is weak
- if `online` helps clearly, `V2a` is real enough to justify the next spend

## 2026-04-01 Corrected 1×H100 Result

The first pod round surfaced a real eval bug in the exploratory runner:

- overlapping sliding windows were scoring the same token more than once

That bug was fixed in the repo-root runner, and the corrected exploratory reference line is now:

- spine: `xsa`
- `model_dim=448`
- `num_layers=9`
- `residual_table_size=49152`
- run suffix: `budget9x448_t49k_fixeval`

The three corrected under-cap runs are:

- `20260401T004320Z_spine_b_seed1337_budget9x448_t49k_fixeval`
- `20260401T004527Z_spine_b_seed2025_budget9x448_t49k_fixeval`
- `20260401T004539Z_spine_b_seed42_budget9x448_t49k_fixeval`

Three-seed summary:

- mean `eval_context.val_bpb = 2.43654`
- mean `eval_static_residual.val_bpb = 2.39399`
- mean `eval_online_residual.val_bpb = 2.33665`
- mean `eval_delta_static_bpb = -0.04255`
- mean `eval_delta_online_bpb = -0.09989`
- std(`eval_delta_online_bpb`) = `0.00359`
- max artifact = `15,874,794` bytes

Interpretation:

- the residual-memory mechanism is real
- the online update is consistently better than the static table
- the exploratory branch is under the decimal `16,000,000` byte cap
- the absolute BPB is still far from submission-grade, so this is a proof-of-direction rather than a submission branch

This is the current canonical `V2a` exploratory config. If we need one line to preserve and graft forward, it is this one.

### Exact Replay

From the record folder on a 1×H100 pod:

```bash
SFW_RUN_SUFFIX=budget9x448_t49k_fixeval \
SFW_MODEL_DIM=448 \
SFW_NUM_LAYERS=9 \
SFW_RESIDUAL_TABLE_SIZE=49152 \
./runpod_spine_b.sh
```

## Promotion

Once you decide which run is worth freezing into the record folder:

```bash
./promote_run.sh runs/<timestamp>_<profile>_seed<seed>
```

## References

- [2026-03-31-spectral-flood-walk-v2a-design.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/plans/2026-03-31-spectral-flood-walk-v2a-design.md)
- [runpod-preflight.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/runpod-preflight.md)
