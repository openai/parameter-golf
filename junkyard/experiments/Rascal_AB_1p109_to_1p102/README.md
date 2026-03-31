# Rascal A/B Lab — 1.109 -> 1.102 Push

Clean A/B workspace sourced from `experiments/Rascal_Stripper` with 4 explicit arms:

- `train_gpt_baseline.py`
- `train_gpt_turbomuon.py`
- `train_gpt_engramlite.py`
- `train_gpt_combo.py`

Goal: isolate effect of each delta vs baseline and test the combined stack.

## Quick Smoke

```bash
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_smoke.sh
```

Default signal profile in `run_ab_smoke.sh`:

- `SEEDS=444` (single seed)
- `ITERATIONS=2200` per arm
- `WARMDOWN_ITERS=0`
- Arms run sequentially: baseline -> turbomuon -> engramlite -> combo

Optional overrides:

```bash
SEEDS="444" ITERATIONS=2200 WARMDOWN_ITERS=0 NPROC=8 \
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_smoke.sh
```

## GB10 Signal Proxy (Single GPU, Low Burn)

Use this when you want fast directional signal before spending time on 8xH100:

```bash
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_gb10_signal.sh
```

Default proxy profile:

- `TORCHRUN_BIN=torchrun` (or set explicitly if your pod has multiple torch installs)
- `NPROC=1`
- `SEEDS=444`
- `ITERATIONS=220` (10% of 2200)
- `TRAIN_BATCH_TOKENS=81920` (~10% of 786432)
- `TRAIN_SEQ_LEN=1024`
- `WARMDOWN_ITERS=0`
- `VAL_LOSS_EVERY=0` (no expensive step-0 validation)
- `SKIP_FINAL_EVAL=1` + `POST_EMA_DIAGNOSTIC=1` (fast single-metric signal)
- `COMPILE_ENABLED=0`
- Arms remain sequential: baseline -> turbomuon -> engramlite -> combo

Optional overrides:

```bash
NPROC=1 SEEDS="444" ITERATIONS=220 TRAIN_BATCH_TOKENS=81920 \
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_gb10_signal.sh
```

## Full 600s A/B

```bash
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_full.sh
```

Optional overrides:

```bash
SEEDS="42 300 444" MAX_WALLCLOCK_SECONDS=600 NPROC=8 \
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_full.sh
```

## Single H100 Step Test (2000/arm)

```bash
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_h100_2000.sh
```

Default profile:

- `NPROC=1`
- `SEEDS=444`
- `ITERATIONS=2000` per arm
- `WARMDOWN_ITERS=0`
- `TRAIN_BATCH_TOKENS=131072`
- Fast signal metric mode: `SKIP_FINAL_EVAL=1`, `POST_EMA_DIAGNOSTIC=1`

## Outputs

Each run writes logs under:

- `experiments/Rascal_AB_1p109_to_1p102/logs/<run_tag>/`

And a machine-readable summary CSV:

- `summary.csv` with `val_bpb_exact`, `delta_vs_baseline`, `gap_vs_target`.

Target is controlled with `TARGET_BPB` (default `1.10200000`).
