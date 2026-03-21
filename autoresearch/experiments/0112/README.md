# Experiment 112

**Date:** 2026-03-19T17:23:37.181428+00:00
**Result:** REVERTED
**val_bpb:** 1.6241
**Artifact size:** 15,397,084 bytes
**Model params:** 28863584
**Last step:** 307
**Pre-quant val_bpb:** 1.6122
**Quantization gap:** 0.0119
**Propose time:** 0.0s
**Train time:** 322.6s

## Change
Reduced train_batch_tokens from 524,288 to 262,144 (halved batch size). On the 1xH100 proxy, this approximately doubles the number of training steps (~720 vs ~360) and doubles the effective LR multiplier (from ~0.30 to ~0.60) due to the warmdown schedule's dependence on step_ms. While each gradient is √2 noisier, the net effect is approximately √2 more total gradient signal per unit wall clock time. The 262K batch (256 sequences) remains well above typical critical batch sizes for language models. Zero impact on model architecture, parameter count, or artifact size. On 8xH100, the model will train for ~28K steps instead of ~14K, with warmdown still well-behaved.

## Diff from previous best
+1 lines / -1 lines (vs current best)
