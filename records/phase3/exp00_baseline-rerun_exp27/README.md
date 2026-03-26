# Baseline Rerun (Exp27 on A100)

## Purpose

Control experiment. Establishes the exact val_bpb of exp27 on A100 hardware with 2 seeds to measure variance. All phase3 experiments are compared against this.

## Protocol

Run twice:
```bash
SEED=42 bash run.sh
SEED=1337 bash run.sh
```

## Results

| Seed | val_bpb | train_time | steps |
|------|---------|------------|-------|
| 42   | TBD     | TBD        | TBD   |
| 1337 | TBD     | TBD        | TBD   |
| **mean** | TBD | | |
| **std**  | TBD | | |
