# SKC + Engram Convergence Fix Plan

Goal: make SKC and Engram causally useful to next-token prediction under the 10-minute 16 MB budget, not just "present with gradients." Three tracks: SKC-only, Engram-only, Combined. All runs use `skc_matrix_v3` harness with `TRAINING_DYNAMICS_ONLY=1`, `SKIP_BUILD_SUBMISSION=1`, `MAX_WALLCLOCK_SECONDS=540`.

## Phase 0: Shared instrumentation

- `SKC_CAUSAL_PROBE=1`: held-out minibatch probe for SKC scale multipliers `{0.0, 0.5, 2.0, 4.0}` with `skc_causal` logs.
- `BRANCH_AMP_LOG=1`: per-layer SKC/MLP/Engram amplitude-to-residual ratios (`amp L{i} ...`).
- `ENG_CAUSAL_PROBE=1`: Engram on/off loss delta via temporary injection disable.
- `ENG_GATE_LOG=1`: gate mean/std/saturation metrics from `EngramHash`.
- Run CSV output: `logs/skc_matrix_<RUN_ID>/probe_summary.csv` with probe deltas, amplitude metrics, and scale summaries.

Acceptance for Phase 0:
- CSV parseable.
- Probe overhead under 5% wall-clock versus probes-off baseline.

## Track A: SKC-only

- Raise SKC residual init via `SKC_RESIDUAL_SCALE_INIT` (default `0.15`).
- Add optional amplitude ramp via `SKC_AMP_RAMP_FRACTION` (default `0.3` for track runs).
- Add SKC structural optimizer group with `SKC_STRUCT_LR_MULT` (default `1.5`).
- Optional head dampening knob: `HEAD_LR_MULT=0.7`.

Decision criteria:
- Use `skc_zero_delta`, `amp_skc/amp_res`, and final loss deltas versus baseline.

## Track B: Engram-only

- Taper bugfix: use threaded `engram_taper_start/end` instead of hard-coded literals.
- Tail taper config for dynamics runs: `ENGRAM_TAPER_START=0.95`, `ENGRAM_TAPER_END=0.99`.
- Sparse update knob: `ENG_WRITE_EVERY` for periodic Engram gradient updates.
- Gate diagnostics via `ENG_GATE_LOG=1`.

Decision criteria:
- Use `eng_zero_delta`, gate saturation stats, final loss versus Engram-off control.

## Track C: Combined

- Add feature-flagged Engram→SKC coupling:
  - `ENG_TO_SKC_MODE=gate`
  - `ENG_TO_SKC_MODE=bias`
- Combine best SKC settings + tail Engram taper + coupling.
- Validate causality with Engram-off ablation from best combined config.

Decision criteria:
- Combined must beat best single-track result and show stronger SKC causal delta.

## Deliverables

- This file: `records/plans/skc_engram_convergence_plan.md`.
- Phase 0 instrumentation diff in `train_gpt_verbose.py`.
- Per-run `probe_summary.csv`.
- Final comparison table (`loss_final`, `step_time_avg_ms`, `skc_zero_delta`, `eng_zero_delta`, `amp_skc/amp_res`).
- Go/no-go recommendation per track.
