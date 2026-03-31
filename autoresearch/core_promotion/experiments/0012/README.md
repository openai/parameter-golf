# Experiment 12

**Date:** 2026-03-19T19:15:36.797510+00:00
**Lane/Stage:** core/promotion
**Result:** KEPT
**val_bpb:** 1.3508
**Artifact size:** 15,833,200 bytes
**Model params:** 17342176
**Last step:** 1061
**Pre-quant val_bpb:** 1.3501
**Quantization gap:** 0.0007
**Eval time:** 15288 ms
**Peak memory:** 12106 MiB
**Gate reason:** improved_val_bpb (1.49810767 -> 1.3508)
**Propose time:** 91.5s
**Train time:** 679.2s

## Change
Increase num_layers from 11 to 12 while reducing model_dim from 512 to 448 to fit the deeper model under the 16MB artifact cap. Depth is the strongest lever (12 layers hit 1.3346 BPB vs 1.4981 at 11 layers) but 12×512 was 20MB — 4MB over budget. At dim=448, estimated params drop ~23% (~17.3M vs 22.6M), projecting ~15.3MB artifact, safely under 16MB while preserving the depth advantage.

## Diff from previous best
Identical to current best
