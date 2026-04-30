# Experiment 4

**Date:** 2026-03-19T18:05:10.528360+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5770
**Artifact size:** 13,002,111 bytes
**Model params:** 24140368
**Last step:** 330
**Pre-quant val_bpb:** 1.5609
**Quantization gap:** 0.0161
**Eval time:** 13397 ms
**Peak memory:** 12703 MiB
**Gate reason:** no_val_bpb_improvement (best=1.5574, got=1.5770)
**Propose time:** 42.9s
**Train time:** 291.3s

## Change
Increase mlp_mult from 2 to 3 to use the ~6MB of available size headroom (current artifact ~9.9MB vs 16MB cap). This increases MLP hidden dimension from 1024 to 1536, adding ~5.2M parameters for better language modeling capacity. Prior experiments confirmed this direction helps (mlp_mult=3 reached 1.5956 in a similar config). The wider MLP should improve BPB while keeping the compressed artifact well under 16MB.

## Diff from previous best
+2 lines / -2 lines (vs current best)
