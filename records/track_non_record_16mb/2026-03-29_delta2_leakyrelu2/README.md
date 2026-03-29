# Session 04 Delta 2: LeakyReLU^2

Date: 2026-03-29
Status: Complete — NEUTRAL (tie)

## Change

Single isolated delta on top of the measured Session 03 anchor (val_bpb 1.12904446).

**MLP activation**: `F.relu(x).square()` -> `F.leaky_relu(x, 0.5).square()`

This is the activation used by the current overall SOTA entry (abaybektursun, 1.1194 BPB).

## Isolation

- `enable_math_sdp` restored to `True` to match the measured anchor state (pre-commit 563700f, math=True)
- No other code changes vs the measured anchor
- Everything else (architecture, schedule, export, eval) identical

## Results

| Metric | Delta 2 | Anchor | Delta |
|--------|---------|--------|-------|
| sliding s64 val_bpb | 1.12904123 | 1.12904446 | -0.00000323 |
| roundtrip val_bpb | 1.15222198 | 1.15247273 | -0.00025075 |
| pre_quant_ema val_bpb | 1.14438546 | 1.14472403 | -0.00033857 |
| steps | 6,511 | 6,564 | -53 |
| step_avg | 92.09 ms | 91.37 ms | +0.72 ms |
| bytes_total | 15,582,968 | 15,751,324 | -168,356 |
| peak memory | 21,274 MiB | 21,274 MiB | 0 |

## Interpretation

- **Verdict: Neutral / tie.** Not a standalone graduating delta.
- Sliding s64 improvement (-0.003 milliBPB) is within noise — effectively zero.
- Pre-quant and roundtrip both improved slightly, suggesting marginally better quantization-friendliness.
- Artifact 168KB smaller — possibly useful when stacking near the 16MB cap.
- Step time +0.72 ms slower, costing 53 steps. The slower throughput roughly cancels the small per-step quality gain under the fixed 600s budget.
- **Keep as a possible stack component** if future deltas need artifact headroom or if combined with a throughput-positive change.
