# Experiment 14

**Date:** 2026-03-19T19:41:16.460573+00:00
**Lane/Stage:** core/promotion
**Result:** REVERTED
**val_bpb:** 1.3547
**Artifact size:** 15,940,390 bytes
**Model params:** 16137952
**Last step:** 1040
**Pre-quant val_bpb:** 1.3541
**Quantization gap:** 0.0006
**Eval time:** 15014 ms
**Peak memory:** 11432 MiB
**Gate reason:** no_val_bpb_improvement (best=1.3457, got=1.3547)
**Propose time:** 113.4s
**Train time:** 717.2s

## Change
Reduce num_kv_heads from 4 to 2 (Priority 1 item 4: KV-head reduction). With dim=448 and head_dim=56, this halves the K and V projection sizes from 448×224 to 448×112 per layer, saving ~1.2M parameters (~1.1MB compressed) across 12 layers. Each KV head now serves 4 query heads instead of 2 (more aggressive GQA). This frees ~1.1MB of artifact budget (from ~15.96MB to ~14.9MB), creating headroom for a future dim increase while preserving the depth advantage. Hypothesis: at this small model scale, the attention capacity from 8 query heads is sufficient even with only 2 shared KV heads, and the parameter savings will not significantly hurt BPB.

## Diff from previous best
+1 lines / -1 lines (vs current best)
