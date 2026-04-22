# Spec 011 seed_42 — execution notes (screening run)

**Run dir:** `runs/011-training-bundle/seed_42/` (local + JP volume `jlxvxeiol4:/workspace/runs/011-training-bundle/seed_42/`)
**Commit:** `893cefd` on `exp/training-bundle` (fixed rotary-reassignment bug from `8d54854`)
**Date:** 2026-04-20
**Pod:** `079ww0a7hwxf29` (8×H100 SXM AP-JP-1, $23.92/hr) — STOPPED
**Mode:** screening — killed at `stopping_early`, no EMA/GPTQ/sliding/TTT (per `feedback_screen_via_training_endpoint_val.md`).

## Status

**Training completed cleanly.** Hit wallclock cap at step 4844 / 596,160 ms. No NaN, no errors, no divergence.

`training_bundle:` config line confirmed at boot with all three knobs:
`wd_taper_start_frac=0.7 wd_taper_final_mult=0.5 muon_grad_power=0.9 qk_gain_init=5.0 qk_gain_per_layer=''`

## Endpoint metrics

| metric | spec 008 (#1736 repro) | spec 011 (bundle) | Δ (011 − 008) |
|---|---|---|---|
| stopping_early step | 4828 | **4844** | +16 |
| stopping_early train_time | 596.18 s | 596.16 s | ≈ equal |
| step-4844 val_loss | — | 2.3429 | — |
| **step-4844 val_bpb (bare)** | **1.0697** (step 4828) | **1.0706** | **+0.0009** |
| diagnostic post-EMA val_bpb | 1.06922 | **1.06994** | +0.00072 |

(Spec 011 ran 16 more training steps in roughly the same wallclock — slightly faster hardware draw than spec 008's pod.)

## Lever-active analysis (research's call to interpret)

**Important caveat for evaluation:** `WD_TAPER_START_FRAC=0.70` is a fraction of `iterations` env var, not of actual stopping_early step. With `ITERATIONS=20000` (default), taper start = step 14000 — **never reached** (stopped at 4844). So WD taper had **zero effect** in this run.

This run effectively measures **GradPower 0.9 alone**, not the bundle. The +0.0009 endpoint Δ is consistent with the spec's "GradPower alone: −0.001 to −0.003" prior being on the optimistic side, or with this being inside seed/bf16 noise (±0.001–0.002). No verdict from execution; research decides.

If research wants the actual bundle measurement, spec 011 needs a re-run with either `ITERATIONS` set to a value where 70% lands inside the wallclock-cap window, or the taper logic needs to key on actual training step / wallclock-budget.

## Matched-step train_loss curve

| step | spec 008 | spec 011 | Δ |
|---|---|---|---|
| 1 | 9.0180 | 9.0180 | 0.0000 |
| 500 | 2.5807 | 2.5863 | +0.0056 |
| 1000 | 2.8105 | 2.8135 | +0.0030 |
| 1500 | 2.6434 | 2.6422 | −0.0012 |
| 2000 | 2.6723 | 2.6725 | +0.0002 |
| 2500 | 2.5580 | 2.5575 | −0.0005 |
| 3000 | 2.5662 | 2.5711 | +0.0049 |
| 3500 | 2.5716 | 2.5747 | +0.0031 |
| 4000 | 2.4095 | 2.4105 | +0.0010 |
| 4000 (val_bpb) | 1.1110 | 1.1125 | +0.0015 |
| 4500 | 2.2803 | 2.2815 | +0.0012 |

Curves track within ±0.006 throughout. No regime change.

## Artifacts

- `train.log` (229 KB) — full training log up through stopping_early + post-EMA val + start of GPTQ (killed mid-GPTQ).
- `launch.out` (empty — torchrun went straight to train.log).
- No `final.json`, no `final_model.ptz`, no checkpoints — screening mode.

## Cost accounting

| item | cost |
|---|---|
| JP pod attempt 1 (NA region miscreate, deleted in seconds) | $0.10 |
| JP pod 8×H100 attempt 1 (DATA_DIR=./data crash) | ~$1.20 |
| JP pod 8×H100 attempt 2 (QKgain shape bug at 8d54854) | ~$1.50 |
| JP pod stop-window | $0 |
| JP 2×H100 smoke @ 893cefd (full 500 steps + serialize crash on `pyminify`) | ~$1.20 |
| JP 8×H100 final run @ 893cefd (full training + post-EMA + ~30s GPTQ before kill) | ~$5.50 |
| **Total spec 011 spend** | **~$9.50** |

## Things that went wrong (memory-worthy)

1. **Spec's launch block had `DATA_DIR=./data`** — wrong; spec 008 used `/workspace/data`. Saved to `project_caseops_data_path.md`.
2. **`pyminify` CLI not preinstalled** — already in `feedback_preflight_deps_and_gpu_clean.md`'s recommended install set; I missed it on first attempt. No memory update needed; just follow the existing one.
3. **Watcher pattern bug** — used `stopping_early at step` but real format is `stopping_early: wallclock_cap`. Memory already documents the correct pattern; my error was not consulting it carefully.
4. **WD taper never engaged** because `WD_TAPER_START_FRAC=0.70` is a fraction of `iterations=20000`, not of real wallclock-cap step. Spec needs a research-side rethink before re-run.

## Handback

Training healthy. Endpoint val_bpb +0.0009 vs spec 008 baseline (within noise). **WD taper did not engage** this run — only GradPower was active. Research to:
- Decide whether +0.0009 is signal or noise (likely noise per per-seed std).
- Decide whether to re-spec the bundle so taper actually fires (rebase taper on wallclock budget or on `iterations_to_run` rather than `iterations`).
- Decide promote / iterate / kill.
