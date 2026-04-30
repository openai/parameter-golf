# Experiment 16

**Date:** 2026-03-19T19:22:02.662033+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5459
**Artifact size:** 15,138,849 bytes
**Model params:** 19988264
**Last step:** 316
**Pre-quant val_bpb:** 1.5434
**Quantization gap:** 0.0025
**Eval time:** 15609 ms
**Peak memory:** 13693 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5459)
**Propose time:** 0.0s
**Train time:** 266.8s

## Change
Reduce num_kv_heads from 4 to 2 to save ~1.5M parameters across 13 layers (K and V projections shrink from 480×240 to 480×120 each), freeing ~1.5MB of artifact budget while preserving full 13-layer depth. GQA with 8 query heads and 2 KV heads (4:1 ratio) is well-established in modern architectures and should maintain attention quality with fewer parameters. This is priority 4 in the research program and the next untried depth-preserving size recovery lever.

## Diff from previous best
+3 lines / -3 lines (vs current best)
