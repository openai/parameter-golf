# Experiment 7

**Date:** 2026-03-19T19:02:27.887342+00:00
**Lane/Stage:** eval_time/discovery
**Result:** KEPT
**val_bpb:** 1.4936
**Artifact size:** 9,602,401 bytes
**Model params:** 17059912
**Last step:** 428
**Pre-quant val_bpb:** 1.4772
**Quantization gap:** 0.0164
**Eval time:** 44573 ms
**Peak memory:** 10239 MiB
**Gate reason:** improved_val_bpb (1.56113154 -> 1.4936)
**Propose time:** 0.0s
**Train time:** 654.2s

## Change
Add post-quantization logit softcap calibration — after temperature search, search over softcap values (coarse grid [15-50] + fine ±2 around best) using a buffer instead of a fixed float. Temperature controls linear scaling; softcap controls nonlinear tanh clipping of extreme logits. After int8 quantization shifts weight magnitudes, the optimal softcap may differ from the training default of 30.0, providing a genuinely orthogonal calibration axis.

## Diff from previous best
Identical to current best
