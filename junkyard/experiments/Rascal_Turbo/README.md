# Rascal_Turbo

Rascal II copy with **TurboMuon-only** injected.

What changed vs Rascal II baseline:

- Newton-Schulz path switched to AOL + Polar coefficients (`NS4` default).
- Added post-NS normalization hook (`MUON_POST_NORM`, default `row_col`).
- No EngramLite changes in this folder.

## One Script

```bash
python3 experiments/Rascal_Turbo/run.py
```

Default behavior:

- 3 seeds: `42,300,444`
- mode: `race`
- `nproc_per_node`: `auto` (uses all visible GPUs)
- wallclock: compute-equivalent to `600s @ 8 GPUs` if not explicitly set
- summary CSV: `experiments/Rascal_Turbo/logs/<run_tag>/summary.csv`

## Common Commands

Race run, 8 GPUs, 3 seeds:

```bash
python3 experiments/Rascal_Turbo/run.py \
  --nproc-per-node 8 \
  --seeds 42,300,444 \
  --mode race
```

Single-GPU signal run (2000-step style):

```bash
python3 experiments/Rascal_Turbo/run.py \
  --nproc-per-node 1 \
  --seeds 444 \
  --mode signal
```

Single-GPU but 8x-equivalent wallclock:

```bash
python3 experiments/Rascal_Turbo/run.py \
  --nproc-per-node 1 \
  --seeds 42,300,444 \
  --mode race
```
