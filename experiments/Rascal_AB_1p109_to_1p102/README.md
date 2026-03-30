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

## Full 600s A/B

```bash
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_full.sh
```

Optional overrides:

```bash
SEEDS="42 300 444" MAX_WALLCLOCK_SECONDS=600 NPROC=8 \
bash experiments/Rascal_AB_1p109_to_1p102/run_ab_full.sh
```

## Outputs

Each run writes logs under:

- `experiments/Rascal_AB_1p109_to_1p102/logs/<run_tag>/`

And a machine-readable summary CSV:

- `summary.csv` with `val_bpb_exact`, `delta_vs_baseline`, `gap_vs_target`.

Target is controlled with `TARGET_BPB` (default `1.10200000`).
