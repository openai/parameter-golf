# Experiment Policy (Active)

## Multi-GPU requirement
- All experiments in this repository are **multi-GPU only**.
- Use `NPROC>=2` for every experimental run (matrix, reruns, compile benchmarks, ablations).
- Single-GPU runs are disallowed unless explicitly approved and marked with `ALLOW_SINGLE_GPU=1`.

## Launcher enforcement
- `run_god_tier_skc.sh` enforces this policy and exits when `NPROC<2` unless `ALLOW_SINGLE_GPU=1` is set.

## Rationale
- Comparability and throughput assumptions in current tracking are based on multi-GPU behavior.
- Single-GPU runs can produce misleading conclusions for both loss and systems-performance comparisons.
