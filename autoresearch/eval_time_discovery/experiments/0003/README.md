# Experiment 3

**Date:** 2026-03-19T18:05:14.757123+00:00
**Lane/Stage:** eval_time/discovery
**Result:** REVERTED
**val_bpb:** 1.6504
**Artifact size:** 8,686,291 bytes
**Model params:** 17059912
**Last step:** 319
**Pre-quant val_bpb:** 1.6144
**Quantization gap:** 0.0360
**Eval time:** 177479 ms
**Peak memory:** 10239 MiB
**Gate reason:** eval_time_exceeded (177479ms > 60000ms)
**Propose time:** 161.9s
**Train time:** 639.7s

## Change
Reduce val_stride from 256 to 64, giving each scored token at least 960 tokens of left context (up from 768) during sliding window evaluation. This trades 4x more eval compute for better-conditioned predictions, directly targeting the eval_time lane's goal of trading evaluation compute for lower val_bpb. The change is a single hyperparameter default; the sliding window evaluation machinery was already active.

## Diff from previous best
+46 lines / -7 lines (vs current best)
