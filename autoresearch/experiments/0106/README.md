# Experiment 106

**Date:** 2026-03-19T16:18:51.810686+00:00
**Result:** REVERTED
**val_bpb:** 1.5805
**Artifact size:** 19,719,597 bytes
**Model params:** 28863584
**Last step:** 317
**Pre-quant val_bpb:** 1.5780
**Quantization gap:** 0.0025
**Propose time:** 0.0s
**Train time:** 279.1s

## Change
Reduced warmdown_iters from 1200 to 300. On the 1xH100 proxy (~300 steps, 180s), the current warmdown_iters=1200 causes the warmdown regime to be active from step 0 with an initial LR multiplier of only 0.25 (warmdown_ms=720s >> training_time=180s). Reducing to 300 makes warmdown_ms match total training time, giving a linear LR decay from 1.0 to 0 over training — a 4x higher average effective learning rate. This should dramatically improve convergence on the short-horizon proxy. On 8xH100, warmdown shortens from ~1200 to ~350 steps, which is still a reasonable cooldown period.

## Diff from previous best
+1 lines / -1 lines (vs current best)
