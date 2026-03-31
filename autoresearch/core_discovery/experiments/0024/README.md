# Experiment 24

**Date:** 2026-03-19T20:13:38.637148+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5691
**Artifact size:** 11,724,799 bytes
**Model params:** 17342176
**Last step:** 318
**Pre-quant val_bpb:** 1.5659
**Quantization gap:** 0.0032
**Eval time:** 15292 ms
**Peak memory:** 12106 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5691)
**Propose time:** 0.0s
**Train time:** 265.0s

## Change
Increase tied_embed_lr from 0.05 to 0.08 to speed up vocabulary representation learning on the short-horizon proxy. The tied embedding matrix (1024×448) must learn both input and output representations from near-zero init (std=0.005) using Adam, while Muon-optimized matrices run at lr=0.085 — the embedding may be the bottleneck for early convergence. Unlike the scalar_lr increase (exp #6, which hurt control parameters already near optimal init), embeddings need to learn rich representations from scratch and should benefit from faster optimization. Zero impact on step time, parameters, or artifact size.

## Diff from previous best
+2 lines / -2 lines (vs current best)
