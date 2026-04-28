# RunPod Rerun Incident Note (2026-04-18)

## What failed
- Rerun batch `rerun_A0_locked_20260418_022227` failed with CUDA OOM at startup.
- Root cause: `TRAIN_BATCH_TOKENS=49152` was forced on 24GB GPUs (free VRAM ~24GB), which is above the safe envelope for this profile.
- Compile bench `bench_reduce_overhead_full_20260418_022637` failed on first run with an Inductor backend compiler assertion under DDP (`NPROC=2`).

## Guardrails added
- `run_god_tier_skc.sh` now clamps `MATRIX_LOCK_BATCH_TOKENS` to `32768` when `FREE_MB_MIN < 30000`, unless `ALLOW_UNSAFE_MATRIX_LOCK=1` is explicitly set.

## Operational checklist (before launching matrix/reruns)
1. Confirm GPU memory class from `nvidia-smi` and choose lock value accordingly.
2. For ~24GB cards, use `TRAIN_BATCH_TOKENS=32768` as baseline.
3. Run a short smoke (`FAST_SMOKE=1`) before full horizon when changing compile knobs.
4. For compile benchmarking, start with `NPROC=1` to validate compile-path stability, then scale to multi-GPU.

## Recommended defaults for this pod class
- `TRAIN_BATCH_TOKENS=32768`
- `MATRIX_LOCK_BATCH_TOKENS=32768`
- `NPROC=2` minimum for all experiments in this repo (multi-GPU policy).
