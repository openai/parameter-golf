# Experiment 9

**Date:** 2026-03-19T18:35:18.377262+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5623
**Artifact size:** 12,884,291 bytes
**Model params:** 20734552
**Last step:** 372
**Pre-quant val_bpb:** 1.5552
**Quantization gap:** 0.0071
**Eval time:** 13467 ms
**Peak memory:** 12296 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4981, got=1.5623)
**Propose time:** 237.8s
**Train time:** 258.5s

## Change
Change warmdown LR schedule from linear to cosine decay. Cosine warmdown keeps the learning rate higher during the initial warmdown phase (e.g., at 25% through warmdown: cosine LR=0.854 vs linear LR=0.75) while still smoothly decaying to 0 at the end. Experiment #7 showed that spending less time in warmdown improves short-horizon convergence; cosine warmdown achieves a similar effect by spending more of the warmdown period at high LR without reducing the warmdown duration itself. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+9 lines / -2 lines (vs current best)
