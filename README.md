# Diffusion Notes For Parameter Golf

This repo is currently set up as a week-1 masked diffusion prototype for the Parameter Golf challenge on Apple Silicon with MLX.

The original upstream challenge README has been preserved in `PARAMETER_GOLF_README.md`.

## Current Status

- Main training script: `train_diffusion.py`
- Scope currently matches the week-1 goal from `DIFFUSION_IMPLEMENTATION_PLAN.md`
- Available baseline configs:
  - `configs/diffusion_tiny.env`
  - `configs/diffusion_local.env`
  - `configs/diffusion_local_smallval.env`
  - `configs/diffusion_scale.env`

## What Is Implemented

- Minimal bidirectional Transformer denoiser for discrete masked diffusion
- Absorbing-mask corruption with timestep-conditioned training
- Synthetic repeated-pattern mode for quick overfit debugging
- FineWeb shard loading on the local `sp1024` dataset subset
- Sample generation by iterative unmasking
- MLX-friendly microbatching for Apple Silicon

## Important Caveat

The current diffusion validation path is only a masked-denoising proxy loss. It is not yet the real Parameter Golf metric (`val_bpb` on the fixed FineWeb validation split).

On 2026-04-08, we found that the diffusion validation corruption settings were coupled to the training corruption settings, which invalidated later cross-run comparisons based on reported diffusion `val_loss`. Those post-week-1 experiment artifacts and claims were removed so the repo reflects the intended week-1 baseline state again.

## Week-1 Outcome

- End-to-end diffusion training works locally
- Synthetic debugging mode is in place
- One-shard FineWeb training is wired up
- Sampling is available for sanity checks
- Proper challenge-aligned evaluation remains the next major step

## Important Files

- Upstream challenge docs: `PARAMETER_GOLF_README.md`
- Diffusion implementation plan: `DIFFUSION_IMPLEMENTATION_PLAN.md`
- Diffusion notes and cleanup log: `EXPERIMENT_LOG.md`
- Main training script: `train_diffusion.py`

## Running The Baseline Local Config

```bash
cd parameter-golf
set -a; source configs/diffusion_local.env; set +a
./.venv/bin/python train_diffusion.py
```

## Running The Tiny Synthetic Config

```bash
cd parameter-golf
set -a; source configs/diffusion_tiny.env; set +a
./.venv/bin/python train_diffusion.py
```
