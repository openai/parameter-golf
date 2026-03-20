# Experiment 15

**Date:** 2026-03-19T19:15:22.622373+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5213
**Artifact size:** 15,257,413 bytes
**Model params:** 19988264
**Last step:** 338
**Pre-quant val_bpb:** 1.5191
**Quantization gap:** 0.0022
**Eval time:** 15595 ms
**Peak memory:** 13693 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5213)
**Propose time:** 65.2s
**Train time:** 334.8s

## Change
Reduce num_kv_heads from 4 to 2 to save ~1.5M parameters (smaller K/V projections across 13 layers), shrinking artifact size by ~1-1.5MB for more headroom under the 16MB cap, while also slightly speeding up each step to get more training steps in the 180s proxy budget. The 8:2 query-to-KV head GQA ratio is widely used in efficient architectures and should preserve most attention quality.

## Diff from previous best
+3 lines / -3 lines (vs current best)
