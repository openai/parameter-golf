# Ablation 2. Batch Size Versus Warmdown

## Goal

Retune the script for the run that actually happens: a 600-second wallclock-limited trainer.

## Why This Is One of the First Ablations

The current default configuration is not reaching its nominal `ITERATIONS=20000`. It stops at `13780` due to wallclock. That means schedule and batch tuning should target the real update budget, not the nominal fixed-step budget.

## Suggested Grid

- `TRAIN_BATCH_TOKENS={131072,262144,524288}`
- `WARMDOWN_ITERS={1200,3000,6000}`

Everything else fixed. Score by final exact roundtrip `val_bpb`.

## Decision Rule

If smaller batch plus longer warmdown beats the current default, then the baseline was overpaying for throughput at the expense of useful optimization.

## Recommendation

Run first
