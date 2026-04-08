# Diffusion Experiment Log

## 2026-04-06 - Week-1 Baseline Complete

- Implemented `train_diffusion.py` as a minimal masked diffusion language-model baseline.
- Added synthetic repeated-pattern mode for quick overfit debugging.
- Added local FineWeb configs for smoke and scale-up runs.
- Added iterative unmasking samples for sanity checks.
- Confirmed end-to-end local training and sampling behavior.

## 2026-04-08 - Repo Cleanup After Validation Bug

- Confirmed that the diffusion validation path reused the same corruption settings as training.
- This meant later cross-run diffusion `val_loss` comparisons were not trustworthy when mask rate, diffusion steps, or schedule changed.
- Removed diffusion logs, sweep configs, suite runners, and result claims that depended on that flawed comparison path.
- Reset the repo surface to the intended end-of-week-1 baseline state.

## Current Interpretation

- The week-1 implementation milestone is complete.
- The next milestone is week 2 from `DIFFUSION_IMPLEMENTATION_PLAN.md`: replace the proxy diffusion validation with a challenge-aligned evaluation path that can report a trustworthy `val_bpb`.
