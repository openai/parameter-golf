# Experiment 12

**Date:** 2026-03-19T20:24:40.488197+00:00
**Lane/Stage:** eval_time/discovery
**Result:** REVERTED
**val_bpb:** N/A (failed)
**Artifact size:** N/A bytes
**Model params:** N/A
**Last step:** N/A
**Pre-quant val_bpb:** N/A
**Quantization gap:** N/A
**Eval time:** N/A ms
**Peak memory:** N/A MiB
**Gate reason:** timeout
**Propose time:** 0.0s
**Train time:** 833.4s
**Error:** timeout

## Change
Add ultra-minimal post-quantization skip-weight scale calibration (Phase 6) with only 2 coarse eval passes [0.92, 1.08] instead of the 9 passes that caused previous attempts (#8, #9, #10) to timeout. After int8 quantization shifts encoder weight magnitudes, the optimal skip connection strength changes; this searches for a global multiplier on skip_weights in-place (no forward pass or graph changes). Conditionally adds 2 more fine-search passes only if improvement is found, for a total of 2-4 passes (~2-4s) vs 9 passes (~9-18s) in prior attempts.

## Diff from previous best
+30 lines / -0 lines (vs current best)
