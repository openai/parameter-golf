# Experiment 12

**Date:** 2026-03-19T18:59:29.522031+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5275
**Artifact size:** 16,192,225 bytes
**Model params:** 22572128
**Last step:** 347
**Pre-quant val_bpb:** 1.5261
**Quantization gap:** 0.0014
**Eval time:** 14735 ms
**Peak memory:** 13389 MiB
**Gate reason:** artifact_over_budget (16192225 > 16000000)
**Propose time:** 0.0s
**Train time:** 264.6s

## Change
Reduce warmdown_iters from 600 to 400 to raise the effective learning rate on the 180s proxy. With ~377 steps at ~477ms/step, warmdown_iters=600 means warmdown_ms (286s) exceeds the 180s budget, so the model never reaches full LR (starting at only ~63% of base). Reducing to 400 raises the initial LR to ~94% of base, giving much more productive learning throughout the short-horizon proxy. On the full 8×H100 10-min run (~4000 steps), 400 warmdown steps is still ~10% of training in cooldown, which is reasonable for smooth end-of-training decay. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+1 lines / -1 lines (vs current best)
