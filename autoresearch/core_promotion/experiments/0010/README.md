# Experiment 10

**Date:** 2026-03-19T18:47:18.912969+00:00
**Lane/Stage:** core/promotion
**Result:** REVERTED
**val_bpb:** 1.3349
**Artifact size:** 20,024,938 bytes
**Model params:** 22572128
**Last step:** 1055
**Pre-quant val_bpb:** 1.3343
**Quantization gap:** 0.0006
**Eval time:** 14661 ms
**Peak memory:** 13389 MiB
**Gate reason:** artifact_over_budget (20024938 > 16000000)
**Propose time:** 43.2s
**Train time:** 729.1s

## Change
Increase num_layers from 11 to 12 to add one more transformer block, continuing the depth-first strategy that has been the strongest single lever (9→10→11 layers each gave clear BPB wins). The extra layer adds ~1.1MB compressed, bringing the estimated artifact to ~14.7MB, still under the 16MB cap with ~1.3MB headroom. Hypothesis: deeper models consistently improve BPB at this width, and the U-Net skip connections help gradient flow even at increased depth.

## Diff from previous best
+1 lines / -1 lines (vs current best)
