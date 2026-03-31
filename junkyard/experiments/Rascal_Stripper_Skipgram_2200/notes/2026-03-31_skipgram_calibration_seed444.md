# Skip-gram calibration (Rascal Stripped, single H100)

Date: 2026-03-31
Run: `experiments/Rascal_Stripper_Skipgram_2200/logs/rascal_stripped_skipgram2200_20260331_014027`
Seed: `444`
Config: `2200` steps, no winddown, `SKIP_FINAL_EVAL=1`, `compile=0`, `train_shards=1`

## Results

| variant | post_ema val_bpb | delta_vs_baseline | step_avg_ms | delta_step_ms |
|---|---:|---:|---:|---:|
| baseline | 1.3264 | +0.0000 | 403.88 | +0.00 |
| skipgram_low (`patterns=1,3`, `mix=0.5`) | 1.3268 | +0.0004 | 405.57 | +1.69 |
| skipgram_high (`patterns=1,3,5;1,2,4;1,4,8`, `mix=1.5`) | 1.3263 | -0.0001 | 412.24 | +8.36 |

## Interpretation
- Low config regressed quality and speed.
- High config gave only a tiny BPB gain (`-0.0001`) while slowing steps materially (`+8.36 ms`, ~2.1%).
- With single seed and this effect size, skip-gram signal is too weak to justify immediate promotion.

## Decision
- Treat skip-gram as **weak/near-noise** under current integration.
- Prioritize next calibrated pack: baseline vs `muon_ns4` vs `loader_cache4` vs combo on single GPU (with at least 4 train shards).
