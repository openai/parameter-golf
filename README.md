# Diffusion Notes For Parameter Golf

This repo now includes the week-3 masked diffusion ablation surface for the Parameter Golf challenge on Apple Silicon with MLX.

The original upstream challenge README has been preserved in `PARAMETER_GOLF_README.md`.

## Current Status

- Main training script: `train_diffusion.py`
- Scope now includes the week-2 validation milestone from `DIFFUSION_IMPLEMENTATION_PLAN.md`
- Available baseline configs:
  - `configs/diffusion_tiny.env`
  - `configs/diffusion_local.env`
  - `configs/diffusion_local_smallval.env`
  - `configs/diffusion_scale.env`
  - `configs/diffusion_week3_local.env`
  - `configs/diffusion_week3_scale.env`
  - `configs/diffusion_week3_scale_long.env`

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
- Week-3 ablation knobs:
  - `MASK_SCHEDULE=uniform|linear|cosine|loglinear`
  - `TRAIN_TIMESTEP_SAMPLING=random|cyclic`
  - `LOSS_REWEIGHTING=none|inverse_mask_rate`
  - `PARAMETERIZATION=x0|xtminus1`
  - `SELF_CONDITIONING=0|1`
  - `SAVE_BEST_CHECKPOINT=0|1`
  - `BEST_CHECKPOINT_METRIC=val_bpb|val_elbo_nats`
  - `SAMPLE_NUM_STEPS_LIST=...`
- Machine-readable `metrics_json:` / `sample_json:` log lines
- Mask-rate proxy-loss bucket reporting
- Best and last checkpoint saving plus a manifest file

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
- Stage A is complete: `linear + cyclic` beat the old local baseline and survived full validation
- Stage B is complete: self-conditioning and inverse-mask-rate reweighting did not beat the promoted Stage-A recipe on the local setup
- Stage C is complete: the clean intended `linear + cyclic` 3000-step rerun is now the strongest confirmed local recipe
- Stage D is complete on the `linear + cyclic` branch: `xtminus1` was screened and rejected against `x0`
- Stage F is complete: optimizer tuning produced a clearly better local recipe, the follow-up dynamic boundary search produced a second optimizer promotion, and the fixed-size scale run improved again
- Latest confirmed promoted full-val metric:
  - recipe: `MASK_SCHEDULE=linear`, `TRAIN_TIMESTEP_SAMPLING=cyclic`, `PARAMETERIZATION=x0`, `SELF_CONDITIONING=0`, `LOSS_REWEIGHTING=none`, `LEARNING_RATE=0.0012`, `WEIGHT_DECAY=0.0`, `BETA2=0.95`, `GRAD_CLIP_NORM=0.2`, `WARMUP_STEPS=20`
  - fixed-size scale context: `TRAIN_SHARDS=2`, `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=32768`, `GRAD_ACCUM_STEPS=4`
  - `val_bpb:2.3249` with `val_elbo_nats:3.9255` over `62021632` validation tokens
  - logfile: `logs/week3_stage_g_scale_20260412_154123/diffusion_week3_scale_diffusion_best_mlx_full_eval.txt`
- The week-3 continuation follow-up pushed the same fixed-size scale branch significantly lower on local-device validation:
  - run: `logs/week3_stage_h_continue_20260412_200615`
  - warm-started from the `P6` best checkpoint, then trained for another `7000` steps
  - best subset checkpoint at step `7000`: `val_bpb:2.1093`
  - final eval from the run log: `val_bpb:2.1158`
  - this run is the strongest achieved local-device quality result so far, but it is not yet the latest confirmed full-val champion because a standalone full eval has not been recorded for it
- The clean 3000-step `linear + cyclic` rerun beat the earlier 3000-step `cosine + cyclic` promotion on both subset and full validation, so the repo default has been switched back to `linear + cyclic`
- The earlier accidental `cosine + random` 3000-step probe remains useful as evidence of training-length headroom, but it is no longer the key Stage C result
- The latest P3 batch remains directly useful, because it was already run on the current best `linear + cyclic` branch:
  - `x0` final subset `val_bpb:2.9320`
  - `xtminus1` final subset `val_bpb:3.2119`
  - the long `xtminus1` follow-up was skipped correctly
- The latest P4 process screen was mostly negative/inconclusive:
  - `32` diffusion steps stayed best among `{16, 32, 64}`
  - `MIN_MASK_RATE=0.05` at `32` steps gave a tiny screen win (`2.910849` best checkpoint) over the previous `1500`-step SOTA, but the margin is negligible
  - `MAX_MASK_RATE=0.95` is not ELBO-valid in the current setup because the final absorbing state must be fully masked
- The promoted P4 long rerun has now completed:
  - recipe: `NUM_DIFFUSION_STEPS=32`, `MIN_MASK_RATE=0.05`, `MAX_MASK_RATE=1.0`
  - best subset checkpoint at `3000` steps: `val_bpb:2.5675`
  - full-val on that best checkpoint: `val_bpb:2.5868`
  - this is slightly worse than the current promoted P2 full-val result (`2.5856`), so the base recipe stays unchanged
- The initial `P5` optimizer batch produced a real promotion:
  - best `1500`-step screen winner: `lr=4e-4`, `wd=0.0`, `beta2=0.95`, `grad_clip_norm=0.3`, `warmup_steps=20`
  - promoted `3000`-step subset best checkpoint: `val_bpb:2.4786`
  - local final subset eval at step `3000`: `val_bpb:2.4952`
  - full-val on the promoted best checkpoint: `val_bpb:2.5005`
  - this beat the prior promoted full-val result (`2.5856`) by about `-0.0851 val_bpb`
- The dynamic `P5` follow-up produced a second optimizer promotion:
  - fresh control reproduced `val_bpb:2.8281` at `1500` steps
  - accepted changes: `grad_clip_norm 0.3 -> 0.2`, then `learning_rate 4e-4 -> 5e-4 -> 7e-4 -> 1.1e-3 -> 1.2e-3`
  - warmup `40` and beta2 `0.92` did not beat the incumbent once the stronger recipe was active
  - promoted `3000`-step subset best checkpoint: `val_bpb:2.3636`
  - local final subset eval at step `3000`: `val_bpb:2.3823`
  - full-val on the promoted best checkpoint: `val_bpb:2.3900`
  - this beat the prior promoted full-val result (`2.5005`) by about `-0.1105 val_bpb`
- The fixed-size `P6` scale run produced a further promotion:
  - same model size as local: `6L x 256d`
  - scale-context training settings: `TRAIN_SHARDS=2`, `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=32768`, `GRAD_ACCUM_STEPS=4`
  - best subset checkpoint at `3000` steps: `val_bpb:2.3102`
  - local final subset eval at step `3000`: `val_bpb:2.3183`
  - full-val on the promoted best checkpoint: `val_bpb:2.3249`
  - this beat the prior promoted full-val result (`2.3900`) by about `-0.0651 val_bpb`
- The longer `P7` continuation run showed that this branch still has substantial headroom on local hardware:
  - best checkpoint at `7000` continuation steps: `val_bpb:2.1093`
  - final logged eval: `val_bpb:2.1158`
  - the run used a weights-only warm start, so it is best interpreted as an achieved-quality reference rather than a clean from-scratch control
- The next week-3 step is a fresh `10000`-step run on this same fixed-size scale recipe rather than reopening local optimizer search

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

## Running The Week-3 Local Config

```bash
cd parameter-golf
set -a; source configs/diffusion_week3_local.env; set +a
./.venv/bin/python train_diffusion.py
```

## Running Standalone Validation

```bash
cd parameter-golf
set -a; source configs/diffusion_local.env; set +a
VAL_MAX_TOKENS=0 ./.venv/bin/python diffusion_eval.py --checkpoint logs/your_run_id_diffusion_mlx.npz
```

## Running The Week-3 Scale Long Recipe

```bash
cd parameter-golf
set -a; source configs/diffusion_week3_scale_long.env; set +a
./.venv/bin/python train_diffusion.py
```

This config is a fresh `10000`-step run of the current best fixed-size scale recipe. It does not warm-start from the stage-H checkpoint.

## Full-Evaluating The Current Best Continued Checkpoint

```bash
cd parameter-golf
bash run_week3_scale_long_full_eval.sh
```

## Running The Tiny Synthetic Config

```bash
cd parameter-golf
set -a; source configs/diffusion_tiny.env; set +a
./.venv/bin/python train_diffusion.py
```
