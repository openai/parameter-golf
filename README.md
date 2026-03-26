This submission captures `A40 Dev Baseline`.

## Summary

Single-GPU NVIDIA A40 development run of the Parameter Golf baseline on Seadragon. This is a non-record submission scaffold intended to preserve the exact log, code snapshot, and artifact-size metrics from the first successful remote CUDA run.

## Key Metrics

- Post-quant exact: `final_int8_zlib_roundtrip_exact val_loss:2.68242198 val_bpb:1.58868139`
- Total submission size int8+zlib: `9123804 bytes`
- Serialized model int8+zlib: `9073403 bytes`
- Counted code size: `50401 bytes`
- Eval time: `54106 ms`

## Files

- `param_golf_baseline_egpu.66621140.out` (training log)
- `train_gpt.py` (self-contained training script snapshot)
- `submission.json` (leaderboard metadata)

## Reproduction Notes

- Track: `track_non_record_16mb`
- Fill in the exact launch command, hardware, and seed details before opening the PR.
- If this is a record submission, include enough logs to demonstrate statistical significance.
