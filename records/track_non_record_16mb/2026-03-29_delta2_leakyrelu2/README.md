# Session 04 Delta 2: LeakyReLU^2

Date: 2026-03-29
Status: Prepared, awaiting run

## Change

Single isolated delta on top of the measured Session 03 anchor (val_bpb 1.12904446).

**MLP activation**: `F.relu(x).square()` -> `F.leaky_relu(x, 0.5).square()`

This is the activation used by the current overall SOTA entry (abaybektursun, 1.1194 BPB).

## Isolation

- `enable_math_sdp` restored to `True` to match the measured anchor state (pre-commit 563700f)
- No other code changes vs the measured anchor
- Everything else (architecture, schedule, export, eval) identical

## Reference

- Session 03 anchor sliding s64 val_bpb: 1.12904446
- Session 03 anchor roundtrip val_bpb: 1.15247273
- Session 03 anchor pre-quant EMA val_bpb: 1.14472403
- Session 03 anchor artifact: 15,751,324 bytes
- Session 03 anchor steps: 6,564, step_avg: 91.37 ms

## Success Criteria

- final_int6_sliding_window_exact val_bpb < 1.12904446
- bytes_total < 16,000,000
- Training behavior comparable to anchor (similar step count and step_avg)

## Results

*Pending run*
