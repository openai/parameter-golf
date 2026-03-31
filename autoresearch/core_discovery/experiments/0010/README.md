# Experiment 10

**Date:** 2026-03-19T18:46:28.459612+00:00
**Lane/Stage:** core/discovery
**Result:** KEPT
**val_bpb:** 1.4967
**Artifact size:** 15,582,419 bytes
**Model params:** 22572128
**Last step:** 367
**Pre-quant val_bpb:** 1.4944
**Quantization gap:** 0.0023
**Eval time:** 14725 ms
**Peak memory:** 13389 MiB
**Gate reason:** improved_val_bpb (1.49810767 -> 1.4967)
**Propose time:** 130.2s
**Train time:** 264.4s

## Change
Reduce warmdown_iters from 800 to 600 to raise the effective learning rate on the 180s proxy. With ~377 steps in 180s, warmdown_ms=800*step_ms vastly exceeds the 180s budget, keeping the LR at only ~47% of base from step 1. Reducing to 600 iters increases the initial LR to ~63%, allowing more productive learning throughout the short-horizon proxy while still preserving smooth end-of-training decay. Zero impact on step time, parameters, or artifact size. On the full 8×H100 10-min run (~4000 steps), warmdown of 600 steps is still reasonable (~15% of training in cooldown).

## Diff from previous best
Identical to current best
