# Bandit Wagon X 9F

Submission-oriented crawler full-run package based on the latest BW12..BW16 signal.

## Architecture

- Tap-off crawler stack (`CRAWLER_TAP_DIM=0`)
- No anchor (`ANCHOR_DIM=0`)
- Deeper floor: `NUM_FLAT_LAYERS=9`
- Crawler core: `NUM_CRAWLER_LAYERS=1`, `CRAWLER_LOOPS=3`, `INST_DIM=32`
- Quant path: naive int6 (no GPTQ in-run), with legal-size guard enabled

## Why this pack exists

This folder is designed to be directly promotable into a submission record if metrics are strong:

- Uses the exact training file to run (`train_gpt.py`)
- Writes required seed logs (`train_seed444.log`, `train_seed300.log`)
- Copies per-seed artifacts with unique filenames
- Emits a metrics TSV for `submission.json` filling
- Enforces the 16MB size limit by default

## Run

```bash
# Primary seed
SEED=444 NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/run.sh

# Confirmation seed
SEED=300 NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/run.sh
```

Optional third seed:

```bash
SEED=4 NPROC_PER_NODE=8 bash records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/run.sh
```

## Outputs

- `train_seed444.log`, `train_seed300.log` (and optional `train_seed4.log`)
- `logs/train_seed<seed>_<timestamp>.log`
- `metrics_seed<seed>.tsv`
- `final_model_seed<seed>.pt`
- `final_model_seed<seed>.int6.ptz`

## Notes

- Default `CRAWLER_QUANT_INT8=0` in `run.sh` is intentional for better chance to stay under 16MB.
- If you want quality-first behavior with higher size risk, override:
  - `CRAWLER_QUANT_INT8=1`
- `submission.json` should be filled only after seed runs complete.
