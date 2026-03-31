# Experiment 21

**Date:** 2026-03-19T20:01:52.906281+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.7663
**Artifact size:** 8,470,984 bytes
**Model params:** 17059912
**Last step:** 339
**Pre-quant val_bpb:** 1.6136
**Quantization gap:** 0.1527
**Eval time:** 11003 ms
**Peak memory:** 10304 MiB
**Gate reason:** quantization_gap_exceeded (0.1527 > 0.0800)
**Propose time:** 559.0s
**Train time:** 281.5s

## Change
Add Exponential Moving Average (EMA) of model weights with decay=0.99 for export. After each optimizer step, update an EMA copy of all parameters via in-place lerp. Before serialization, swap the model's weights with the EMA weights. EMA smooths out gradient noise from late-stage updates, producing weights with smaller dynamic range and fewer outliers — both of which improve int8 quantization fidelity and should reduce the quantization gap (currently 0.0339 BPB). Compute overhead is negligible (~0.15ms/step for 17M-param in-place lerp). Memory overhead is ~69MB for the EMA buffer (trivial on H100).

## Diff from previous best
+10 lines / -0 lines (vs current best)
