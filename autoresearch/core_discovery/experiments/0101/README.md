# Experiment 101

**Date:** 2026-03-19T19:33:33.871011+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5281
**Artifact size:** 11,928,623 bytes
**Model params:** 17342176
**Last step:** 335
**Pre-quant val_bpb:** 1.5256
**Quantization gap:** 0.0025
**Eval time:** 15285 ms
**Peak memory:** 12105 MiB
**Gate reason:** no_val_bpb_improvement (best=1.3508, got=1.5281)
**Propose time:** 45.1s
**Train time:** 326.0s

## Change
Reduce warmdown_iters from 600 to 550 to test whether a slightly shorter warmdown phase improves convergence. The warmdown sweep (1200→800→600) has consistently improved BPB, but 400 was too aggressive. 550 probes the sweet spot between the current best (600) and the known-bad (400), giving ~8% more steps at full learning rate while maintaining smooth end-of-training decay. Zero impact on model size, parameters, or step time — purely a schedule change with no artifact risk.

## Diff from previous best
+1 lines / -1 lines (vs current best)
