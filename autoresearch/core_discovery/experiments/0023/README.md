# Experiment 23

**Date:** 2026-03-19T20:07:01.529913+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5440
**Artifact size:** 11,410,167 bytes
**Model params:** 17342176
**Last step:** 314
**Pre-quant val_bpb:** 1.5402
**Quantization gap:** 0.0038
**Eval time:** 15283 ms
**Peak memory:** 12106 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5440)
**Propose time:** 0.0s
**Train time:** 264.1s

## Change
Increase warmdown_iters from 600 to 700 to test whether a slightly longer warmdown phase improves convergence on the 180s proxy. The warmdown sweep so far shows 800→1.5247, 600→1.4967 (best), 550→1.5245, 400→1.5174 — all values below 600 hurt. Testing 700 brackets the optimum from above to determine whether the true minimum is at 600 or slightly higher. On the 180s proxy (~377 steps), warmdown_iters=700 gives warmdown_ms=334s, so the starting LR is ~54% of base (vs ~63% at 600). On the full 600s run (~1061 steps), warmdown starts at step 361 (vs 461 at 600). Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+2 lines / -2 lines (vs current best)
