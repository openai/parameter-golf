# Experiment Plan

Goal: make the budget-run candidate more submittable by adding one small empirical improvement over the current valid baseline.

Original valid baseline:

```text
seed: 42
train_shards: 86
steps: 3975
valid_sliding_bpb: 1.11043153
total_bytes: 15,987,889
```

Best current result:

```text
seed: 42
train_shards: 86
qk_gain_init: 4.5
steps: 4271
valid_sliding_bpb: 1.10743376
total_bytes: 15,987,195
```

## Completed Paid Run

Run one targeted variant:

```bash
QK_GAIN_INIT=4.5 \
RUN_ID=sp4096_1xh100_seed42_86shards_qk45 \
./scripts/parameter_golf/run_sp4096_budget86_existing_data.sh
```

Why this first:

Outcome:

- Improved valid sliding BPB by about `0.0030`.
- Required repeated artifact coarsening of `ve_shared.embed.weight` to fit the decimal 16MB cap.
- Keep as the current best budget candidate.

## Follow-up Queue

Run these one at a time only if spending more RunPod budget:

```bash
QK_GAIN_INIT=4.5 SEED=1337 RUN_ID=sp4096_1xh100_seed1337_86shards_qk45 ./scripts/parameter_golf/run_sp4096_budget86_existing_data.sh
MATRIX_LR=0.018 RUN_ID=sp4096_1xh100_seed42_86shards_mlr018 ./scripts/parameter_golf/run_sp4096_budget86_existing_data.sh
MUON_WD=0.09 EMBED_WD=0.09 RUN_ID=sp4096_1xh100_seed42_86shards_wd09 ./scripts/parameter_golf/run_sp4096_budget86_existing_data.sh
QK_GAIN_INIT=5.0 RUN_ID=sp4096_1xh100_seed42_86shards_qk50 ./scripts/parameter_golf/run_sp4096_budget86_existing_data.sh
```

## Submission Bar

Submit as a non-record only after one of these is true:

- A single-knob variant beats the baseline and a second seed confirms it is not obvious noise.
- Or the folder is framed purely as a budget reproduction/engineering note, not as a new ML improvement.
