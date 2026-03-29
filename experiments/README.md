# Parameter Golf Experiments

This folder holds isolated mini-projects that fork the baseline [train_gpt.py](../train_gpt.py) in one narrow area at a time so results stay attributable.

Current experiments:

- `exp01_mixed_export`
  - Same model and training path as baseline.
  - Changes only the export policy: large MLP matrices are packed to int4, while the rest of the model follows the baseline-style int8 or float passthrough path.
- `exp02_factored_embeddings`
  - Same export path as baseline.
  - Changes only the embedding/head path by introducing a factorized token embedding with a projection into `model_dim`.

## Baseline Comparison

Run the baseline from [train_gpt.py](../train_gpt.py) and each experiment with the same data path, tokenizer, and local smoke settings. Example:

```bash
cd parameter-golf
RUN_ID=baseline_smoke \
ITERATIONS=50 \
WARMUP_STEPS=0 \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=524288 \
uv run python train_gpt.py
```

Then run an experiment from its own folder:

```bash
cd parameter-golf/experiments/exp01_mixed_export
RUN_ID=exp01_smoke \
ITERATIONS=50 \
WARMUP_STEPS=0 \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=524288 \
uv run python train_gpt.py
```

## Optuna

The shared [run_optuna.py](./run_optuna.py) script is intended for local proxy studies, not leaderboard claims. It is a good fit for:

- export-policy sweeps
- a small number of schedule knobs
- one or two architectural knobs with cheap smoke settings

It is a bad fit for:

- brute-forcing many seeds for leaderboard claims
- unconstrained full-budget sweeps on the official 10-minute target
- tuning tokenizer/data changes without additional correctness checks

Recommended use is:

1. tune on a cheap proxy budget
2. lock a narrow candidate range
3. rerun promising settings manually against the baseline
