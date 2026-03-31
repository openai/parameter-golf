# Experiment 115

**Date:** 2026-03-19T18:02:09.457915+00:00
**Result:** KEPT
**val_bpb:** 1.5487
**Artifact size:** 15,991,541 bytes
**Model params:** 28863584
**Last step:** 322
**Pre-quant val_bpb:** 1.5363
**Quantization gap:** 0.0124
**Propose time:** 508.3s
**Train time:** 280.9s

## Change
Reduced logit_softcap from 30.0 to 20.0. With only 1024 vocabulary tokens, logits of ~14 nats suffice for 99.99% confident predictions; a softcap of 30 is unnecessarily loose. The tighter cap of 20 provides implicit regularization on output logit magnitudes without limiting model expressiveness (tanh(x/20) stays nearly linear for |x|<10, and can still output ~15.2 effective logit units). This should improve quantization-friendliness by keeping output-facing weight magnitudes smaller, while the gradient at practical logit values remains healthy (d/dx[tanh(x/20)] > 0.45 at x=15). Zero impact on parameter count, step time, or artifact size.

## Diff from previous best
Identical to current best
