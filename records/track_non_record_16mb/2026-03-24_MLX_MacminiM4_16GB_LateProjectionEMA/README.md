# Non-record Submission: MLX Late Projection EMA Finalizer on Mac mini M4 16GB

This folder captures a non-record Apple Silicon submission copied from the published run package in `frido22/low_vram_institute`, using only the verified best eligible package:

- Source repo run package: `output/runs/2026_03_24_run_0033`
- Run title: `Late Projection EMA Finalizer`

This is not an official 8xH100 record submission. It is a Mac mini M4 16GB MLX run submitted under `records/track_non_record_16mb` because the hardware differs from the main leaderboard setting.

## Verified Published Result

- Run ID: `2026_03_24_run_0033`
- Hardware: `Mac mini M4 16GB`
- Framework: `MLX`
- Final exact post-quant metric: `final_int8_zlib_roundtrip_exact val_bpb: 1.56720003`
- Final exact post-quant loss: `3.53539760`
- Last in-training validation before stop: `val_bpb: 1.5717`, `val_loss: 3.5455`
- Stop condition: `wallclock_cap` at step `899`
- Train time: `534904 ms`
- Published runtime: `602.147632 s`
- Int8+zlib model bytes: `15888695`
- Total artifact bytes: `15962372`

These values were verified against the published run package and repo state in `frido22/low_vram_institute`, including `state/ledger.jsonl`, `output/reports/history.csv`, `output/runs/2026_03_24_run_0033/submission.json`, `output/runs/2026_03_24_run_0033/artifact_size.json`, and `output/runs/2026_03_24_run_0033/run.log`.

## Technique Summary

This run keeps a late EMA over only the projection matrices that still end up int8-quantized after the first quant-aware roundtrip, then reapplies the exact final roundtrip before save. The goal is to improve the score that actually matters for the compressed artifact without increasing artifact size or slowing the hot training path for most of the run.

## Why This Is Non-record

- The artifact is under the `16,000,000` byte limit.
- The run was executed on Apple Silicon with MLX on a `Mac mini M4 16GB`.
- It is therefore presented as a hardware-specific non-record submission rather than an official 8xH100 leaderboard result.

## Included Files

- `train_gpt_mlx.py` - exact published MLX training script from the verified run package
- `train.log` - exact published training log from the verified run package
- `requirements.txt` - published dependency snapshot from the verified run package
- `submission.json` - metadata for this non-record submission

No other files from the source repo are included in this submission folder.
