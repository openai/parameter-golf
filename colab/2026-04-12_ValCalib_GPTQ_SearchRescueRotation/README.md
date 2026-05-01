# Colab Experiment: 2026-04-12_ValCalib_GPTQ_SearchRescueRotation

This folder is a Colab-friendly fork of the March 25 accepted record replica:

- benchmark reference: `colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`
- record source: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`

The goal is to test export-time ideas from `QuantExport3` on the March 25 stack without changing the benchmark surface.

## What Changed

Unlike the March 25 Colab folder, this folder contains a full local copy of the record trainer so it can be patched safely. The original record folder is not modified.

New opt-in export features:

- `EXPORT_SEARCH_ENABLED`
- `SEARCH_ROTATION_OPTIONS`
- `SEARCH_MIXED_PRECISION_BUDGETS`
- `SEARCH_MIXED_PRECISION_MAX_TENSORS`
- `SEARCH_GPTQ_BLOCK_SIZES`
- `SEARCH_MAX_FRONTIER_EVALS`
- `ROTATION_AWARE_ENABLED`
- `ROTATION_BLOCK_SIZE`
- `HESSIAN_DAMPING`
- `GPTQ_ACCEPT_MSE_RATIO`
- `MIXED_PRECISION_EXTRA_BUDGET_BYTES`
- `MIXED_PRECISION_MAX_TENSORS`
- `GPTQ_AR_CALIB_NUM_SEQS`
- `GPTQ_AR_CALIB_SEQ_LEN`
- `GPTQ_AR_CALIB_BATCH_SIZE`
- `GPTQ_AR_CALIB_TEMPERATURE`

Default behavior preserves the March 25 Colab replica export path:

```bash
EXPORT_SEARCH_ENABLED=0 bash run.sh
```

## What To Compare

Use the same metric for every run:

- `final_int6_sliding_window_s64_exact`

Useful supporting log lines:

- `export_candidate`
- `export_roundtrip`
- `export_selected`
- `artifact_size_table`
- `search_frontier_table`
- `roundtrip_quality_table`

Generated search tables:

- `logs/artifact_size_table.tsv`
- `logs/search_frontier.tsv`
- `logs/roundtrip_quality_table.tsv`

## Colab Usage

```bash
git clone https://github.com/IanniMuliterno/parameter-golf.git
cd parameter-golf/colab/2026-04-12_ValCalib_GPTQ_SearchRescueRotation
INSTALL_DEPS=1 bash run.sh
```

## Recommended Smoke Matrix

These smoke commands use a smaller autoregressive GPTQ calibration set so Colab T4 runs finish export quickly. For a record-aligned benchmark run, remove the `GPTQ_AR_CALIB_*` overrides and keep the rest of the comparison surface unchanged.

Baseline parity:

```bash
RUN_ID=baseline_parity \
ITERATIONS=300 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
GPTQ_AR_CALIB_NUM_SEQS=8 \
GPTQ_AR_CALIB_SEQ_LEN=512 \
GPTQ_AR_CALIB_BATCH_SIZE=4 \
EXPORT_SEARCH_ENABLED=0 \
bash run.sh
```

Search infrastructure only:

```bash
RUN_ID=search_infra_only \
ITERATIONS=300 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
GPTQ_AR_CALIB_NUM_SEQS=8 \
GPTQ_AR_CALIB_SEQ_LEN=512 \
GPTQ_AR_CALIB_BATCH_SIZE=4 \
EXPORT_SEARCH_ENABLED=1 \
SEARCH_ROTATION_OPTIONS=0 \
SEARCH_MIXED_PRECISION_BUDGETS=0 \
SEARCH_MIXED_PRECISION_MAX_TENSORS=0 \
SEARCH_GPTQ_BLOCK_SIZES=128 \
bash run.sh
```

Mixed precision only:

```bash
RUN_ID=mixed_precision_only \
ITERATIONS=300 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
GPTQ_AR_CALIB_NUM_SEQS=8 \
GPTQ_AR_CALIB_SEQ_LEN=512 \
GPTQ_AR_CALIB_BATCH_SIZE=4 \
EXPORT_SEARCH_ENABLED=1 \
SEARCH_ROTATION_OPTIONS=0 \
SEARCH_MIXED_PRECISION_BUDGETS=0,65536,131072,262144 \
SEARCH_MIXED_PRECISION_MAX_TENSORS=0,1,2 \
SEARCH_GPTQ_BLOCK_SIZES=128 \
bash run.sh
```

Rotation only:

```bash
RUN_ID=rotation_only \
ITERATIONS=300 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
GPTQ_AR_CALIB_NUM_SEQS=8 \
GPTQ_AR_CALIB_SEQ_LEN=512 \
GPTQ_AR_CALIB_BATCH_SIZE=4 \
EXPORT_SEARCH_ENABLED=1 \
ROTATION_AWARE_ENABLED=1 \
SEARCH_ROTATION_OPTIONS=0,1 \
SEARCH_MIXED_PRECISION_BUDGETS=0 \
SEARCH_MIXED_PRECISION_MAX_TENSORS=0 \
SEARCH_GPTQ_BLOCK_SIZES=64,128 \
HESSIAN_DAMPING=0.03 \
GPTQ_ACCEPT_MSE_RATIO=1.03 \
bash run.sh
```

Mixed precision plus rotation:

```bash
RUN_ID=mixed_precision_plus_rotation \
ITERATIONS=300 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
GPTQ_AR_CALIB_NUM_SEQS=8 \
GPTQ_AR_CALIB_SEQ_LEN=512 \
GPTQ_AR_CALIB_BATCH_SIZE=4 \
EXPORT_SEARCH_ENABLED=1 \
ROTATION_AWARE_ENABLED=1 \
SEARCH_ROTATION_OPTIONS=0,1 \
SEARCH_MIXED_PRECISION_BUDGETS=0,65536,131072,262144 \
SEARCH_MIXED_PRECISION_MAX_TENSORS=0,1,2 \
SEARCH_GPTQ_BLOCK_SIZES=64,128 \
HESSIAN_DAMPING=0.03 \
GPTQ_ACCEPT_MSE_RATIO=1.03 \
SEARCH_MAX_FRONTIER_EVALS=6 \
bash run.sh
```

## Notes

- Keep `NUM_LAYERS` at the default `11` when comparing to the March 25 stack.
- Keep `BIGRAM_VOCAB_SIZE=3072` and `BIGRAM_DIM=112` for benchmark alignment.
- Search runs can be much slower because frontier candidates are evaluated by roundtrip validation before final artifact selection.
- The Colab numbers are directional only; the original accepted record ran on 8xH100.
