# Infra

Compute launch scripts and environment notes for running the training pipeline on different clusters.

| File | Purpose |
|------|---------|
| [`runpod_baseline.py`](runpod_baseline.py) | Runpod pod launcher |
| [`modal_baseline.py`](modal_baseline.py) | Modal container launcher |
| [`RUNPOD_BASELINE.md`](RUNPOD_BASELINE.md) | Runpod setup notes |

Conventions:
- Each cluster script should accept the same env-var interface (`TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`, `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `SEED`, `RUN_ID`) so an experiment config can be dispatched to any cluster unchanged.
- Cluster-specific setup (module loads, sbatch headers, etc.) goes in this dir.
