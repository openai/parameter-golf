# Experiment 22

**Date:** 2026-03-19T20:02:16.568314+00:00
**Lane/Stage:** core/discovery
**Result:** REVERTED
**val_bpb:** 1.5331
**Artifact size:** 12,527,061 bytes
**Model params:** 18585152
**Last step:** 317
**Pre-quant val_bpb:** 1.5301
**Quantization gap:** 0.0030
**Eval time:** 15552 ms
**Peak memory:** 12979 MiB
**Gate reason:** no_val_bpb_improvement (best=1.4967, got=1.5331)
**Propose time:** 0.0s
**Train time:** 284.9s

## Change
Increase model_dim from 448 to 464 to add ~1.2M parameters (17.3M→18.6M) for more model capacity. dim=464 divides cleanly into 8 heads (head_dim=58, even for RoPE) with 4 KV heads. The current config produces ~12MB artifacts on the 180s proxy, leaving ~4MB headroom under the 16MB cap. On the 600s validated run, estimated artifact size ~13MB (ratio 0.70 bytes/param), still well under budget. All Priority 1 (schedule) and Priority 2 (LR) refinements have been exhausted on the current config with regressions, so this tests Priority 3: whether a modest width increase can improve BPB via extra expressiveness, despite ~7% fewer steps from the increased FLOPs.

## Diff from previous best
+2 lines / -2 lines (vs current best)
