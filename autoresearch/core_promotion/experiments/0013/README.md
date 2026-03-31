# Experiment 13

**Date:** 2026-03-19T19:28:27.513660+00:00
**Lane/Stage:** core/promotion
**Result:** KEPT
**val_bpb:** 1.3457
**Artifact size:** 15,964,527 bytes
**Model params:** 17342176
**Last step:** 1076
**Pre-quant val_bpb:** 1.3450
**Quantization gap:** 0.0007
**Eval time:** 15317 ms
**Peak memory:** 12106 MiB
**Gate reason:** improved_val_bpb (1.35077344 -> 1.3457)
**Propose time:** 89.7s
**Train time:** 679.2s

## Change
Reduce warmdown_iters from 600 to 500 to give ~100 more training steps at full learning rate in the 600s proxy. The warmdown→BPB trend has been consistently positive (1200→800→600 each improved), and at 600s budget with warmdown_iters=600, approximately 50% of training is in cooldown. Reducing to 500 lowers this to ~42%, allocating more effective learning time while still maintaining smooth end-of-training decay. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
Identical to current best
