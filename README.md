# Diffusion Notes For Parameter Golf

This repo is currently set up as a week-2 masked diffusion baseline for the Parameter Golf challenge on Apple Silicon with MLX.

The original upstream challenge README has been preserved in `PARAMETER_GOLF_README.md`.

## Current Status

- Main training script: `train_diffusion.py`
- Scope now includes the week-2 validation milestone from `DIFFUSION_IMPLEMENTATION_PLAN.md`
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
- Deterministic diffusion validation over the fixed `fineweb_val_*` split
- D3PM-style ELBO lower-bound reporting as `val_elbo_nats`, `val_bits_per_token`, and `val_bpb`
- Standalone checkpoint evaluation via `diffusion_eval.py`

## Validation Behavior

- `proxy_loss` is retained as debugging telemetry only.
- The main comparison metric is now `val_bpb`, derived from a deterministic ELBO-style lower bound.
- The default local and scale configs now validate on a subset of the validation shard for faster iteration.
- `VAL_MAX_TOKENS` controls that subset size. Use `VAL_MAX_TOKENS=0` for challenge-aligned full-split numbers.

## Milestone Status

- End-to-end diffusion training works locally
- Synthetic debugging mode is in place
- One-shard FineWeb training is wired up
- Sampling is available for sanity checks
- Challenge-aligned validation is wired into training and standalone evaluation
- Week 2 is complete: full-shard standalone eval was rerun successfully and reproduced the same final metric
- Latest confirmed full-val metric from `logs/diffusion_local_diffusion_mlx_full_eval.txt`:
  - `val_bpb:3.0502` with `val_elbo_nats:5.1501` over `62021632` validation tokens

## Important Files

- Upstream challenge docs: `PARAMETER_GOLF_README.md`
- Diffusion implementation plan: `DIFFUSION_IMPLEMENTATION_PLAN.md`
- Diffusion notes and cleanup log: `EXPERIMENT_LOG.md`
- Main training script: `train_diffusion.py`
- Standalone evaluator: `diffusion_eval.py`
- Validation/objective helpers: `diffusion_objectives.py`, `validation_common.py`

## Running The Baseline Local Config

```bash
cd parameter-golf
set -a; source configs/diffusion_local.env; set +a
./.venv/bin/python train_diffusion.py
```

This local config intentionally uses a validation subset. For a full-shard run, override with `VAL_MAX_TOKENS=0`.

## Running Standalone Validation

```bash
cd parameter-golf
set -a; source configs/diffusion_local.env; set +a
VAL_MAX_TOKENS=0 ./.venv/bin/python diffusion_eval.py --checkpoint logs/your_run_id_diffusion_mlx.npz
```

## Running The Tiny Synthetic Config

```bash
cd parameter-golf
set -a; source configs/diffusion_tiny.env; set +a
./.venv/bin/python train_diffusion.py
```
