# PR 414 — Record: 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 (val_bpb: 1.1233)

**Author:** Tianhao Wu (signalrush)
**Claimed BPB:** val_bpb 1.12278022 (val_loss 1.89576235), seed 1337 submitted; 3-seed mean 1.1233 (std 0.0005)
**Artifact size:** 15,555,017 bytes (seed 1337); 15.55 MB mean
**Seeds:** 1337, 42, 2024

## Files retrieved
- `records__track_10min_16mb__2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233__README.md`
- `records__track_10min_16mb__2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233__submission.json`
- `records__track_10min_16mb__2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233__train_gpt.py`

## Claimed changes (from README, verbatim)

> val_bpb: 1.1233 (sliding window stride=64, 3-seed mean) | 15.55 MB (mean) | 8xH100 SXM, 600s
>
> Key Innovations Over PR #374. Two novel post-training optimizations plus training hyperparameter tuning on top of PR #374's architecture:
>
> | Change | PR #374 | This | Impact |
> | GPTQ-lite | Fixed clip (row max) | 5 clip percentiles per row, pick min MSE | -0.0006 BPB (zero training cost) |
> | EMA | None (Tight SWA only) | EMA decay=0.997 every step | -0.0006 BPB (smoother averaging) |
> | Warmdown | 3000 | 3500 | -0.0002 BPB |
> | Late QAT threshold | 0.1 | 0.15 | -0.0001 BPB (earlier fake quant, smaller quant gap) |
> | Total | 1.1246 | 1.1233 | -0.0013 BPB |
>
> GPTQ-lite: Per-Layer Optimal Clip Percentile Search. Instead of using the row maximum for int6 quantization scale, we try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row and pick the one minimizing reconstruction MSE. Applied during post-training quantization with zero training cost.
>
> EMA Weight Averaging. Exponential moving average (decay=0.997) maintained every training step, applied before quantization. Stacks with Tight SWA — EMA provides continuous smoothing while SWA captures discrete checkpoints during warmdown.
>
> Per-seed: 1337 → 7101 steps, 1.8958, 1.1228, 15.56 MB. 42 → ~7100, 1.8972, 1.1236, 15.54 MB. 2024 → ~7100, 1.8971, 1.1236, 15.59 MB. Mean 1.1233, std 0.0005. Submitted: seed 1337 (best).
>
> Run Command: SEED=1337 bash eval/eval.sh
