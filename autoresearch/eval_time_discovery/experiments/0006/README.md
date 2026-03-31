# Experiment 6

**Date:** 2026-03-19T18:46:19.179926+00:00
**Lane/Stage:** eval_time/discovery
**Result:** REVERTED
**val_bpb:** 1.5738
**Artifact size:** 9,058,040 bytes
**Model params:** 17059912
**Last step:** 360
**Pre-quant val_bpb:** 1.5493
**Quantization gap:** 0.0245
**Eval time:** 44536 ms
**Peak memory:** 10239 MiB
**Gate reason:** no_val_bpb_improvement (best=1.5611, got=1.5738)
**Propose time:** 289.1s
**Train time:** 679.0s

## Change
Add eval-time logit softcap calibration after temperature search. After int8 quantization the logit distribution shifts; temperature corrects the linear regime but softcap controls the non-linear tanh clipping, so calibrating it post-quantization is an independent degree of freedom. Coarse+fine grid search over softcap values using fast no-overlap eval adds ~17 cheap eval passes.

## Diff from previous best
+15 lines / -1 lines (vs current best)
