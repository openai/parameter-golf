# Best Repeatable 10-Min Recipe (as of 2026-04-18)

Purpose: preserve the strongest observed training behavior and make reruns repeatable.

## Launcher
- Script: `scratch/run_best_comp10m_recipe.sh`
- Core compile path: `COMPILE_MODE=max-autotune-no-cudagraphs`, `COMPILE_TARGET=full`

## Fixed knobs
- `NPROC=2`
- `TRAIN_BATCH_TOKENS=32768`
- `MATRIX_LOCK_BATCH_TOKENS=32768`
- `MAX_WALLCLOCK_SECONDS=540`
- `DDP_FIND_UNUSED_PARAMETERS=1`

## Eval/export settings (enabled)
- `TRAINING_DYNAMICS_ONLY=0`
- `DIAGNOSTICS_ENABLED=0`
- `ROUNDTRIP_LOGIT_AUDIT=1`
- `ROUNDTRIP_LOGIT_AUDIT_TOKENS=1024`
- `ROUNDTRIP_LOGIT_AUDIT_ENFORCE=0`
- `EXPORT_PARITY_HARNESS=1`

## Why this is frozen
- Recent evidence run completed cleanly at high step count with strong loss descent.
- Compile mode remained stable where other compile modes failed under DDP.
- This is now the baseline to beat; future experiments should branch from this recipe.
