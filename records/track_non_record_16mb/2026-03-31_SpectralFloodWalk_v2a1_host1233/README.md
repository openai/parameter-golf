# Spectral Flood Walk v2a.1 Host Graft

This record folder packages the next step after the exploratory `V2a` runner:

- keep the proven `1.1233` host stack intact
- graft only the residual overlay mechanism on top
- start with **online-only** residual tables
- explicitly verify that the corrected distribution remains normalized

The host lineage is:

- [2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233)

## What Changed

The host training stack stays the same:

- `11L / 512d / MLP3x`
- Muon + AdamW split optimizer
- EMA
- GPTQ-lite int6 export
- sliding-window eval with `stride=64`

The graft adds only:

```python
logits = base_logits + delta_logits
probs = torch.softmax(logits, dim=-1)
```

where `delta_logits` comes from:

- `ResidualBasis`
- `ResidualRouter`
- `ResidualTables`

The first run starts with:

- empty tables in the artifact
- `ResidualBasis` initialized from embedding SVD
- online updates only after scored positions

## Why This Version Matters

Hashed context mechanisms are under heavy scrutiny in the competition right now.

This branch is designed to be explicit about the legality difference:

- it never adjusts only the realized token probability
- it always forms a full corrected logit vector before softmax
- it logs `prob_sum_max_deviation` so we can show the distribution remains normalized

## Target Hardware

This folder is versioned for **1×H100 exploratory runs**.

Defaults:

- `SFW_TARGET_HARDWARE=1xH100 exploratory`
- `SFW_TARGET_GPU_COUNT=1`
- `SFW_NPROC_PER_NODE=1`

## Fast Workflow

From a RunPod shell in `/workspace/parameter-golf`:

```bash
cd records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233
./runpod_preflight.sh
./runpod_smoke.sh
./runpod_full.sh
python3 ../../../tools/summarize_v2a1_host1233_runs.py runs_host1233/*
```

For the `8×H100` ladder, use the dedicated wrappers:

```bash
./runpod_8x_smoke.sh
./runpod_8x_signcheck.sh
./runpod_8x_fullval.sh
```

`runpod_smoke.sh` intentionally uses a reduced validation slice via `VAL_TOKEN_LIMIT=524288` by default so it stays a real smoke test instead of re-running the full single-GPU host eval. Override that env var upward if you want a heavier smoke.

## Fresh Box Workflow

From your local machine, if the pod starts essentially empty:

```bash
cd /Users/kennethmalloy/Local\ Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233
./prepare_fresh_pod.sh <host> <port> root
```

By default this:

- uses direct TCP SSH with `~/.ssh/id_runpod`
- updates `/workspace/parameter-golf` in place to `origin/main` if it is already a git checkout
- clones `openai/parameter-golf` into `/workspace/parameter-golf` only if needed
- overlays only the `V2a.1` graft files
- installs `requirements.txt` if core deps are missing
- fetches cached `sp1024` data with `80` train shards
- runs `py_compile` plus the residual unit tests on the pod

Useful overrides:

- `SFW_POD_SSH_KEY=/path/to/key`
- `SFW_BOOTSTRAP_TRAIN_SHARDS=80`
- `SFW_REMOTE_BASE_REF=main`
- `SFW_DRY_RUN=1` to print the commands without executing them

If the full run is promising:

```bash
./runpod_three_seeds.sh
python3 ../../../tools/summarize_v2a1_host1233_runs.py runs_host1233/*
```

## Expected Outputs

Each run writes:

- `train.log`
- `result.json`
- `residual_artifact.npz`
- `command.sh`
- `notes.txt`

The important `result.json` fields are:

- `eval_context.val_bpb`
- `eval_online_residual.val_bpb`
- `eval_online_residual.delta_bpb`
- `eval_online_residual.prob_sum_max_deviation`
- `artifact.total_bytes`

## Decision Rule

- if `delta_bpb` is negative and stable, the graft is real
- if `prob_sum_max_deviation` is effectively zero, the normalization story is clean
- if the delta collapses on the strong host, we stop and pivot again

## Verified Exploratory Results

Two single-H100 exploratory runs are now copied into `runs_host1233/`:

- `20260401T033200Z_smoke_seed1337_debugtiny`
  - reduced smoke profile with `VAL_TOKEN_LIMIT=524288`
  - `context val_bpb = 4.02841`
  - `online_residual val_bpb = 4.01089`
  - `delta_bpb = -0.01751`
  - `prob_sum_max_deviation = 2.38e-07`
  - `artifact.total_bytes = 5,068,510`

- `20260401T033600Z_full_seed1337_full4m`
  - single-H100 exploratory full run with `VAL_TOKEN_LIMIT=4194304`
  - training stopped at `step 732` on the `600s` cap
  - host `int6` sliding-window `context val_bpb = 3.51007`
  - `online_residual val_bpb = 3.41063`
  - `delta_bpb = -0.09944`
  - `prob_sum_max_deviation = 7.75e-07`
  - `residual table resident MB = 13.5`
  - `artifact.total_bytes = 6,804,759`

The main takeaway from the stronger run is that the residual overlay preserved exact normalization and kept almost the same `~0.10 BPB` improvement we saw in the lightweight `V2a` branch, but now on the `1.1233` host lineage rather than on the custom exploratory runner.

## Current Defaults

The wrappers default to the host-like configuration:

- `NUM_LAYERS=11`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3`
- `XSA_LAST_N=4`
- `RESIDUAL_RANK=16`
- `RESIDUAL_TABLE_SIZE=49152`
- `RESIDUAL_ORDERS=1,2,3,4`

## References

- [2026-03-31-spectral-flood-walk-v2a-design.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/plans/2026-03-31-spectral-flood-walk-v2a-design.md)
- [runpod-preflight.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/runpod-preflight.md)
