# Experiment 18

**Date:** 2026-03-19T19:27:53.632772+00:00
**Lane/Stage:** storage/discovery
**Result:** KEPT
**val_bpb:** 1.6510
**Artifact size:** 8,789,475 bytes
**Model params:** 17059912
**Last step:** 332
**Pre-quant val_bpb:** 1.6192
**Quantization gap:** 0.0318
**Eval time:** 11008 ms
**Peak memory:** 10240 MiB
**Gate reason:** improved_val_bpb (1.6561 -> 1.6510)
**Propose time:** 0.0s
**Train time:** 277.7s

## Change
Fix quantization-dequantization precision mismatch by simulating fp16 scale roundtrip in the MSE-optimal int8 search. Previously, the search computed MSE using fp32 scales but stored them as fp16, causing the search to find clip fractions optimized for an unrealizable precision — and the final int8 values were quantized with a different scale than what dequantization uses. Now the search, quantization, and dequantization all use the same fp16-rounded scale, eliminating systematic reconstruction bias at zero extra artifact cost.

## Diff from previous best
Identical to current best
