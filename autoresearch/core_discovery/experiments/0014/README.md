# Experiment 14

**Date:** 2026-03-19T19:08:17.982761+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5029
**Artifact size:** 15,327,973 bytes
**Model params:** 22572128
**Last step:** 343
**Pre-quant val_bpb:** 1.5003
**Quantization gap:** 0.0026
**Eval time:** 14713 ms
**Peak memory:** 13389 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5029)
**Propose time:** 0.0s
**Train time:** 264.3s

## Change
Increase matrix_lr from 0.08 to 0.09 to test whether Muon optimizer convergence speed can be further improved on the short-horizon proxy. The progression 0.04→0.06→0.08 has shown consistent BPB improvements; this tests whether the optimum lies above 0.08. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+1 lines / -1 lines (vs current best)
