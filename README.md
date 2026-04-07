# Diffusion Notes For Parameter Golf

This repo is currently being used to prototype a simple masked diffusion language model for the Parameter Golf challenge on Apple Silicon with MLX.

The original upstream challenge README has been preserved in `PARAMETER_GOLF_README.md`.

## Current Status

- Main training script: `train_diffusion.py`
- Running logbook: `EXPERIMENT_LOG.md`
- Current best local recipe:
  - `MODEL_DIM=384`
  - `TRAIN_SEQ_LEN=256`
  - `NUM_LAYERS=6`
  - `NUM_DIFFUSION_STEPS=16`
  - `TRAIN_BATCH_TOKENS=32768`
  - `LEARNING_RATE=5e-4`
  - `MASK_SCHEDULE=cosine`
  - `MAX_MASK_RATE=0.65`

## Best Result So Far

- Best long local run: `val_loss=3.9445`
- Run: `logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l01_mask065_diffusion.txt`
- Key takeaway: reducing corruption severity has been the strongest lever so far, stronger than increasing depth or sequence length.

## Daily Progress

### 2026-04-06

- Implemented the week-1 MLX diffusion baseline in `train_diffusion.py`.
- Added synthetic overfit mode, local FineWeb configs, smaller validation support, and Mac-friendly MLX batching.
- Verified end-to-end training on synthetic data.
- Verified one-shard FineWeb learning and built the first experiment log.
- Early result: the baseline learned, but plateaued around `val_loss ~6.0`.

### 2026-04-07

- Ran an overnight 10-experiment suite across width, depth, sequence length, diffusion steps, and mask rate.
- Main outcome: lowering `MAX_MASK_RATE` from `1.0` to `0.8` helped a lot; `384d` also helped when paired with easier corruption.
- Ran a focused follow-up suite around the best recipe.
- Main outcome: lowering `MAX_MASK_RATE` again to `0.7` and increasing `TRAIN_BATCH_TOKENS` to `32768` both helped; together they gave a big jump.
- Ran longer 2000-step mask-rate comparisons.
- Main outcome: `MAX_MASK_RATE=0.65` beat `0.70` and `0.75`, reaching `val_loss=3.9445`.

## What We Learned

- Lower mask rate has consistently improved optimization and final validation loss.
- More batch tokens helped materially.
- `16` diffusion steps beat `32` in this local setup.
- `12` diffusion steps were only a minor improvement over `16`.
- Wider models helped when the corruption process was made easier.
- Extra depth did not help.
- `TRAIN_SEQ_LEN=512` underperformed `256` at the current batch budget.
- The `2`-shard check at `MAX_MASK_RATE=0.70` was roughly tied with the `1`-shard run, which suggests the recipe is reasonably robust.

## Important Files

- Upstream challenge docs: `PARAMETER_GOLF_README.md`
- Diffusion implementation plan: `DIFFUSION_IMPLEMENTATION_PLAN.md`
- Experiment history: `EXPERIMENT_LOG.md`
- Best local config: `configs/diffusion_local_best.env`
- Longer single-run config: `configs/diffusion_local_long.env`
- Longrun suite: `configs/longrun/manifest.txt`

## Running The Best Current Config

```bash
cd parameter-golf
set -a; source configs/diffusion_local_best.env; set +a
./.venv/bin/python train_diffusion.py | tee logs/diffusion_local_best_console.txt
```

## Running The Longrun Suite

```bash
cd parameter-golf
./scripts/run_longrun_diffusion_suite.sh
```

All meaningful experiment outcomes should be appended to `EXPERIMENT_LOG.md` so future work has full context.
