# Experiment 9

**Date:** 2026-03-19T18:32:14.342686+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6623
**Artifact size:** 8,624,762 bytes
**Model params:** 17059912
**Last step:** 332
**Pre-quant val_bpb:** 1.6220
**Quantization gap:** 0.0403
**Eval time:** 11006 ms
**Peak memory:** 10239 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 251.8s

## Change
Activate the unused muon_weight_decay=0.02 hyperparameter by applying decoupled weight decay to Muon (matrix) parameters after each optimizer step. The decay factor per step is `muon_weight_decay * matrix_lr * warmdown_scale`, which shrinks weight magnitudes → tighter per-row dynamic range → smaller int8 quantization steps → better post-quantization BPB. This directly targets the 0.0287 quantization gap observed in experiment #7.

## Diff from previous best
+8 lines / -0 lines (vs current best)
