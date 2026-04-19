# Analysis scripts for spec 006 artifacts

Self-contained analyses over a dense-checkpoint run (spec 006 or equivalent).

**Expected inputs** (paths absolute, adjust per run):
- `CKPT_DIR` — directory with `ckpt_event_step{100,200,...,4500}.pt` plus auto-emitted `ckpt_{warmdown_start,pre_recurrence,final_pre_ema,final_post_ema}_step*.pt`
- `TRAIN_LOG` — `train.log` with per-5-step `train_loss` + per-layer grad norms + per-100-step val_loss

**Expected outputs** (written to the same run dir under `analysis/`):
- `delta_matrix.csv` — per-window per-layer rel-movement
- `lr_normalized.csv` — movement divided by mid-window LR
- `loss_curves.{csv,png}` — train + val overlay
- `grad_norms.{csv,png}` — per-layer grad-norm time series
- `loop_differential.csv` — loop vs non-loop movement ratio over time
- `dynamics_report.md` — prose summary with flagged anomalies

## Scripts

- `parse_train_log.py` — extract (step, train_loss, val_loss, per-layer grad norms) into tidy CSV
- `windowed_weight_delta.py` — adapted from runs/005/weight_delta.py; runs over all consecutive checkpoint pairs
- `lr_schedule.py` — reproduces the step-based LR schedule (warmup + linear warmdown) for normalization
- `plots.py` — matplotlib figures (loss, grad-norm heatmap, movement-vs-LR scatter)
- `run_all.sh` — top-level orchestrator; given a run dir, produces all outputs

## Status
Skeletons written 2026-04-20 during spec 006's training run. Will be validated + debugged against spec 006's artifacts when they land.
