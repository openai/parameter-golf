# Experiment 003 — TTT LR × epochs sweep on the winning filter

**Date:** TBD (Day 2)
**Hypothesis:** With a smaller adapt surface (e.g. `scales`), the optimal `TTT_LR` is higher than 0.005 and the optimal `TTT_EPOCHS` may differ from 3. Selective-TTT changes the loss landscape — the SOTA's tuning was for `all`.
**Baseline:** The winning filter from Experiment 002 at SOTA defaults (LR=0.005, EPOCHS=3)
**Cost:** ~5×3 grid = 15 cells × ~2 min × $6/hr = ~$3 per single seed

## Setup

Run only after Experiment 002 identifies a winning filter. Use the saved checkpoint from Day 1.

## Grid

```
TTT_LR     ∈ {0.002, 0.005, 0.010, 0.020, 0.040}
TTT_EPOCHS ∈ {1, 3, 5}
```

15 cells. Each ~2 min on 2×H100. Total ~30 min at $6/hr ≈ $3.

For `scales` filter specifically — since the gradient flow is now restricted to ~38K floats — much higher LRs (e.g. 0.04, 0.1) may work; we extend the LR grid upward only after seeing whether smaller LRs already saturate.

## Commands

```bash
WINNER_FILTER=scales   # set from experiment 002

for LR in 0.002 0.005 0.010 0.020 0.040; do
  for EP in 1 3 5; do
    TAG="lr${LR//./_}_ep${EP}"
    TTT_ENABLED=1 TTT_PARAM_FILTER=$WINNER_FILTER \
      TTT_LR=$LR TTT_EPOCHS=$EP \
      SEED=42 \
      LOAD_CHECKPOINT=$CKPT \
      RUN_ID=opus_e003_${TAG} \
      torchrun --standalone --nproc_per_node=2 \
        Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/003_${TAG}.log
  done
done
```

## Result

Fill in:

| LR \ EPOCHS | 1 | 3 | 5 |
|-------------|---|---|---|
| 0.002       |   |   |   |
| 0.005       |   |   |   |
| 0.010       |   |   |   |
| 0.020       |   |   |   |
| 0.040       |   |   |   |

## Decision

Pick the LR×EPOCHS combo with the lowest `val_bpb_ttt`. If multiple are within 0.0003: pick the one with the lower epochs count (faster eval, more headroom for the 600s eval budget).

If best cell beats Experiment 002's `f_all` baseline by **≥0.003 nats**: promote to Experiment 004 (chunk-size sweep on the winning config).
