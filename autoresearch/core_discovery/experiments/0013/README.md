# Experiment 13

**Date:** 2026-03-19T19:03:54.179671+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5148
**Artifact size:** 15,503,862 bytes
**Model params:** 22572128
**Last step:** 337
**Pre-quant val_bpb:** 1.5126
**Quantization gap:** 0.0022
**Eval time:** 14725 ms
**Peak memory:** 13389 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5148)
**Propose time:** 0.0s
**Train time:** 263.8s

## Change
Increase matrix_lr from 0.08 to 0.09 to test whether the Muon optimizer benefits from a slightly higher learning rate on the short-horizon proxy. The progression 0.04→0.06→0.08 has consistently improved BPB; this extends that sweep by one step. Zero impact on artifact size, parameter count, or step time — purely a hyperparameter change. If the trend continues, we get a free BPB improvement; if it regresses, we've bracketed the optimum at 0.08-0.09.

## Diff from previous best
+1 lines / -1 lines (vs current best)
