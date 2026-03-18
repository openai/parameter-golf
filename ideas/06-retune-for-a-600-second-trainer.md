# 6. Retune for a 600-Second Trainer

## Core Thesis

The current schedule and batch settings are probably wrong for the run that actually happens, because the script never reaches its nominal iteration count before the wallclock stop.

## What Bottleneck It Attacks

This attacks update budget and schedule shape under the real constraint. The defaults are:

- `ITERATIONS=20000`
- `WARMDOWN_ITERS=1200`
- `TRAIN_BATCH_TOKENS=524288`
- wallclock stop at `600` seconds

Relevant code:

- hyperparameters: `train_gpt.py:49-88`
- wallclock-aware LR multiplier: `train_gpt.py:924-932`
- wallclock stop logic: `train_gpt.py:1050-1055`

## Why It Should Improve `val_bpb`

The baseline stops at step `13780`, not `20000`. That means many intuitions you would use for a fixed-step run do not directly apply here. The script is effectively a 600-second trainer, so batch size and warmdown should be tuned for the number of updates you can actually afford within that time, not for a target iteration count you never reach.

This is not a generic tuning suggestion. It is unusually important here because the training loop is explicitly wallclock-capped, and the current run is still improving near the stop.

## Expected Effect

- Training speed: per-step speed changes with batch, total wallclock fixed
- Evaluation speed: unchanged
- Compressed artifact size: unchanged

## Difficulty

1/5

## Rule-Risk

1/5

## Smallest Decisive Experiment

Run a compact grid at fixed 600 seconds:

- `TRAIN_BATCH_TOKENS={131072,262144,524288}`
- `WARMDOWN_ITERS={1200,3000,6000}`

Measure final exact roundtrip `val_bpb`, not pre-quant validation.

## Recommendation Bucket

Baseline script improvement
