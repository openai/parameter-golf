# Train-Only MTP Experiment

This folder contains an MLX experiment runner built from a copy of `train_gpt_mlx.py`.

Goal:

- test training-only multi-token prediction heads
- vary `MTP_LOSS_WEIGHT`
- vary how long MTP remains active during training
- exclude MTP heads from the exported artifact so the final roundtrip score is still measured on the base model only

## What Changed

Compared with the baseline MLX script, this runner adds:

- `mtp_heads` auxiliary language-model heads
- `MTP_NUM_HEADS`
- `MTP_LOSS_WEIGHT`
- `MTP_ACTIVE_UNTIL_FRACTION`
- sequential sweep support via:
  - `SWEEP_MTP_LOSS_WEIGHTS`
  - `SWEEP_MTP_ACTIVE_UNTIL_FRACTIONS`
- export-time exclusion of `mtp_heads.*`
- a summary TSV written after each run

## Recommended Grid

From `OPTIM_IDEAS.md`, the most relevant first-pass sweep is:

- `MTP_NUM_HEADS=1`
- `MTP_LOSS_WEIGHT in {0.03, 0.07, 0.10, 0.15}`
- `MTP_ACTIVE_UNTIL_FRACTION in {0.6, 0.8, 1.0}`

`1.0` is included as a control even though the main hypothesis is that MTP should usually be disabled before the endgame.

## Local Smoke Command

```bash
RUN_ID=mlx_mtp_sweep \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MTP_NUM_HEADS=1 \
SWEEP_MTP_LOSS_WEIGHTS=0.03,0.07,0.10,0.15 \
SWEEP_MTP_ACTIVE_UNTIL_FRACTIONS=0.6,0.8,1.0 \
python3 records/track_10min_16mb/2026-04-04_train_only_MTP_experiment/train_gpt_mlx_mtp_sweep.py
```

## Outputs

Outputs go to this folder's `logs/` directory by default.

For each run you get:

- one per-run log file
- one exported model artifact without MTP heads
- one quantized roundtrip artifact
- one summary TSV named `<RUN_ID>_summary.tsv`

The key comparison field is:

- `roundtrip_val_bpb`
