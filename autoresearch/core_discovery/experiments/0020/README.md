# Experiment 20

**Date:** 2026-03-19T19:52:20.072057+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5341
**Artifact size:** 11,584,235 bytes
**Model params:** 17342176
**Last step:** 328
**Pre-quant val_bpb:** 1.5303
**Quantization gap:** 0.0038
**Eval time:** 15280 ms
**Peak memory:** 12106 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5341)
**Propose time:** 0.0s
**Train time:** 263.6s

## Change
Reduce matrix_lr from 0.085 to 0.075 to test whether the Muon optimizer's optimal learning rate is below the current default. The progression 0.04→0.06→0.08 showed consistent BPB improvements, but 0.09 was worse (experiments #13/#14). The current default of 0.085 sits between 0.08 and 0.09 but hasn't been compared against slightly lower values on this 12×448 architecture. Testing 0.075 helps bracket the optimum from below — if it improves BPB, the current default is too high; if it regresses, we've confirmed ~0.08-0.085 is near optimal. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+1 lines / -1 lines (vs current best)
