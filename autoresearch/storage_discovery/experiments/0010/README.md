# Experiment 10

**Date:** 2026-03-19T18:36:26.455983+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6645
**Artifact size:** 8,666,357 bytes
**Model params:** 17059912
**Last step:** 328
**Pre-quant val_bpb:** 1.6234
**Quantization gap:** 0.0411
**Eval time:** 11006 ms
**Peak memory:** 10239 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 0.0s
**Train time:** 249.1s

## Change
Connect the existing but unused muon_weight_decay=0.02 hyperparameter to the Muon optimizer by adding decoupled weight decay (p -= lr*wd*p) to its step function. This shrinks matrix weight magnitudes during training, reducing per-row dynamic range and outliers that waste int8 quantization levels. The tighter weight distributions should yield smaller quantization scales, better int8 fidelity, and a narrower quantization gap — directly targeting the post-export val_bpb metric without any parameter count change or step-time overhead.

## Diff from previous best
+7 lines / -2 lines (vs current best)
