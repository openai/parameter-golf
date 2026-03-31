# Rerun of PR #1120 (Rascal) — seed 1337

## Summary

Reran the `train_gpt.py` from PR #1120's submission commit (`39ed402`) with `SKIP_GPTQ=1` on 8xH100 SXM (GCP).

The pre-quant sliding window result is **1.11350** vs the published **1.10979** (seed 300) / mean **1.1099**.

## Environment

- 8x H100 80GB SXM (GCP `a3-highgpu-8g`)
- Driver 565.57.01, Python 3.12, PyTorch 2.9.1+cu128
- `NCCL_NET=Socket`, `SKIP_GPTQ=1`
- Command: `SKIP_GPTQ=1 torchrun --standalone --nproc_per_node=8 train_gpt.py`

## Results

| Metric | Published (seed 300) | Rerun (seed 1337) | Delta |
|--------|---------------------|-------------------|-------|
| `final_sliding_window_exact val_bpb` | **1.10979099** | **1.11350327** | **+0.00371** |
| `final_sliding_window_exact val_loss` | 1.87383064 | 1.88009865 | +0.00627 |
| Steps | 6593 | 6881 | +288 |
| step_avg | ~91ms | 87.2ms | -3.8ms |

## Notes

- The rerun is on seed 1337 (not seed 300), so some seed variance is expected. Typical seed variance for this architecture is ~0.0005 BPP (std).
- The **+0.00371 BPP gap** is 7x larger than typical seed variance.
- The rerun gets MORE training steps (6881 vs 6593) due to faster step time (87.2ms vs ~91ms), yet the result is significantly worse.
- The submitted `train_gpt.py` does not contain quantization code — it only outputs `final_model.pt` and `final_sliding_window_exact`. The `int6+zstd` quantization and `final_int6_roundtrip` metrics visible in the published seed logs appear to be produced by an external runner, not by `train_gpt.py` itself.
- The reported `final_sliding_window_exact` metric is measured on the **pre-quant model** (before any int6/int8 quantization).
