# Experiment 11

**Date:** 2026-03-19T19:00:11.390148+00:00
**Lane/Stage:** core/promotion
**Result:** REVERTED
**val_bpb:** 1.3346
**Artifact size:** 20,023,320 bytes
**Model params:** 22572128
**Last step:** 1062
**Pre-quant val_bpb:** 1.3339
**Quantization gap:** 0.0007
**Eval time:** 14657 ms
**Peak memory:** 13389 MiB
**Gate reason:** artifact_over_budget (20023320 > 16000000)
**Propose time:** 0.0s
**Train time:** 679.2s

## Change
Increase num_layers from 11 to 12 to add one more transformer block, continuing the depth-over-width strategy that has been the strongest single lever (each prior depth increase was a clear win). At ~1.8MB per layer in int8, the artifact should land around ~15.4MB, within the 16MB cap. Hypothesis: deeper models capture more hierarchical language structure per parameter, and the small quantization gaps (~0.004) at this architecture scale indicate the extra layer will quantize well.

## Diff from previous best
+1 lines / -1 lines (vs current best)
