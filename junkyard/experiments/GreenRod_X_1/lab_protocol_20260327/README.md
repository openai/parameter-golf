# 1-GPU A/B Lab Protocol (GreenRod)

This folder follows strict protocol:
- no modification of existing source files;
- each arm/seed run gets a fresh copy of `train_gpt.py`;
- env snapshot and logs are stored per run;
- promotion is automatic against control.

## Files
- `concept_arms.tsv`: arm definitions (edit by making a new copy if you want a new experiment set).
- `run_ab_1gpu_promote.sh`: runner.

## Quick start
```bash
cd /home/frosty40/parameter-golf-lab/experiments/GreenRod_X_1/lab_protocol_20260327
bash run_ab_1gpu_promote.sh
```

## Typical economical profile
```bash
SEEDS=1337,1338 \
NPROC_PER_NODE=1 \
MAX_WALLCLOCK_SECONDS=180 \
VAL_LOSS_EVERY=200 \
SKIP_FINAL_EVAL=1 \
PROMOTE_DELTA=0.010 \
bash run_ab_1gpu_promote.sh
```

## Promotion rule
Candidate is promoted only if it beats control by at least `PROMOTE_DELTA`
on every tested seed for cap `val_bpb`.
