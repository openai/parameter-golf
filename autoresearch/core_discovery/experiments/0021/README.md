# Experiment 21

**Date:** 2026-03-19T19:57:49.572647+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5441
**Artifact size:** 11,681,275 bytes
**Model params:** 17342176
**Last step:** 313
**Pre-quant val_bpb:** 1.5407
**Quantization gap:** 0.0034
**Eval time:** 15283 ms
**Peak memory:** 12106 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5441)
**Propose time:** 0.0s
**Train time:** 267.0s

## Change
Increase logit_softcap from 30.0 to 50.0 to reduce gradient compression through the tanh nonlinearity at the output layer. With softcap=30, even moderate logits (e.g., z=15) suffer 21% gradient attenuation (sech²(0.5)=0.786); at softcap=50 this drops to only 9% (sech²(0.3)=0.914). This allows the model to learn sharper, more confident output distributions faster on the short-horizon proxy, potentially improving BPB. Zero impact on parameter count, artifact size, or step time.

## Diff from previous best
+2 lines / -2 lines (vs current best)
