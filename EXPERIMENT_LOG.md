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

## 2026-04-08 - Week-2 Validation Milestone Complete

- Implemented the week-2 validation pipeline with:
  - deterministic proxy loss
  - ELBO-style `val_elbo_nats`
  - tokenizer-aware `val_bits_per_token`
  - tokenizer-aware `val_bpb`
- Added standalone checkpoint evaluation in `diffusion_eval.py`.
- Added toy and math sanity tests for the absorbing-mask schedule, posterior weighting, determinism, and byte accounting.
- Confirmed the full validation path starts cleanly on the full `fineweb_val_*` split and writes its own eval log file.
- Completed repeated full-shard evaluation on `logs/diffusion_local_diffusion_mlx.npz` and observed the same final metric on rerun.
- The recorded full-eval logfile is `logs/diffusion_local_diffusion_mlx_full_eval.txt`.
- The confirmed full-validation metric is:
  - `final_diffusion_eval proxy_loss:5.5301 val_elbo_nats:5.1501 val_bits_per_token:7.4301 val_bpb:3.0502 tokens:62021632 corruption_samples:4 timestep_samples:1`

## Current Interpretation

- The week-1 implementation milestone is complete.
- The week-2 validation milestone is complete.
- The next milestone is week 3 from `DIFFUSION_IMPLEMENTATION_PLAN.md`: improve the baseline recipe now that evaluation is trustworthy enough for ablation work.
