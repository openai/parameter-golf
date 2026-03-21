# Experiment 5

**Date:** 2026-03-19T18:10:44.745724+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5743
**Artifact size:** 13,075,403 bytes
**Model params:** 24140368
**Last step:** 330
**Pre-quant val_bpb:** 1.5590
**Quantization gap:** 0.0153
**Eval time:** 13405 ms
**Peak memory:** 12703 MiB
**Gate reason:** no_val_bpb_improvement (best=1.5574, got=1.5743)
**Propose time:** 0.0s
**Train time:** 261.2s

## Change
Increase mlp_mult from 2 to 3 to add 50% more MLP hidden capacity per layer (1024→1536 hidden dim). The model is only ~9.9MB of 16MB budget, so the extra ~2.6MB of parameters is well within the artifact cap. MLP layers are the primary knowledge/pattern storage in transformers, and prior work showed mlp_mult=3 improved BPB when paired with appropriate learning rates.

## Diff from previous best
+2 lines / -2 lines (vs current best)
