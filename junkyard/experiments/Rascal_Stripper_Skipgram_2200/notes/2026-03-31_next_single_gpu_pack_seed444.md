# Next Single-GPU Pack (Seed 444)

Date: 2026-03-31
Run: `experiments/Rascal_Stripper_Skipgram_2200/logs/rascal_next_single_gpu_20260331_024750`
Seed: `444`
Config: `1200` steps, `train_batch_tokens=786432`, `train_shards=4`, `compile=1`, `SKIP_FINAL_EVAL=1`

## Results

| case | post_ema val_bpb | delta_vs_baseline | step_avg_ms | delta_step_ms |
|---|---:|---:|---:|---:|
| baseline | 1.3112 | +0.0000 | 786.00 | +0.00 |
| muon_ns4 | 1.3138 | +0.0026 | 786.23 | +0.23 |
| loader_cache4 | 1.3101 | -0.0011 | 782.59 | -3.41 |
| combo_ns4_cache4 | 1.3136 | +0.0024 | 782.39 | -3.61 |

## Interpretation
- `loader_cache4` is the only clean improvement on both speed and quality.
- `muon_ns4` regressed both metrics in this stripped setup.
- Combining `muon_ns4` with cache4 preserves speed but loses quality.

## Next Hypothesis
- The win is loader behavior, not Muon.
- With `COPRIME_MAX_LOADED_SHARDS=4` fixed, tuning loader hold/stride can yield additional speed without giving back BPB.
- Priority order:
  1. `cache4_hold96` and `cache4_hold128` to test reload-pressure reduction.
  2. `cache4_stride1_hold64` to test whether tighter stride improves short-run BPB.

## Next Run
- Script: `experiments/Rascal_Stripper_Skipgram_2200/run_loader_refine_single_gpu.sh`
- Runner: `experiments/Rascal_Stripper_Skipgram_2200/run_loader_refine_single_gpu.py`
