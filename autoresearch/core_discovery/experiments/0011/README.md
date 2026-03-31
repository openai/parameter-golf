# Experiment 11

**Date:** 2026-03-19T18:53:03.101991+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5174
**Artifact size:** 16,203,227 bytes
**Model params:** 22572128
**Last step:** 350
**Pre-quant val_bpb:** 1.5158
**Quantization gap:** 0.0016
**Eval time:** 14725 ms
**Peak memory:** 13389 MiB
**Gate reason:** artifact_over_budget (16203227 > 16000000)
**Propose time:** 121.3s
**Train time:** 265.1s

## Change
Reduce warmdown_iters from 600 to 400 to increase effective learning rate on the 180s proxy. With ~377 steps in 180s, warmdown_iters=600 starts the LR at only ~63% of base (since warmdown_ms=286s exceeds the 180s budget, the model is always in warmdown). With warmdown_iters=400, warmdown_ms=191s, so the LR starts at ~94% of base, giving the model substantially more productive learning time at high LR. The progression 1200→800→600 has consistently improved BPB; this continues that trend. On the full 8×H100 10-min run (~4000 steps), 400 warmdown iters means ~10% of training in cooldown, which is reasonable. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+1 lines / -1 lines (vs current best)
