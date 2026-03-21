# Experiment 15

**Date:** 2026-03-19T19:55:07.204736+00:00
**Lane/Stage:** core/promotion
**Result:** REVERTED
**val_bpb:** 1.3467
**Artifact size:** 16,005,230 bytes
**Model params:** 16137952
**Last step:** 1112
**Pre-quant val_bpb:** 1.3460
**Quantization gap:** 0.0007
**Eval time:** 14962 ms
**Peak memory:** 11431 MiB
**Gate reason:** artifact_over_budget (16005230 > 16000000)
**Propose time:** 0.0s
**Train time:** 678.1s

## Change
Reduce num_kv_heads from 4 to 2 to use a 4:1 GQA ratio (8 query heads sharing 2 KV heads), saving ~1.2M parameters (~1MB compressed artifact size). This is the highest-priority untried change (Priority 1 item 4). The 4:1 GQA ratio is well-established in modern LLMs (Llama 2 70B uses the same ratio) and typically maintains attention quality while significantly reducing parameter count. The ~1MB savings brings the artifact well under the 16MB cap, creating headroom for future width/depth increases. Hypothesis: attention quality is maintained because 4 query heads per KV group can still learn diverse attention patterns, while the reduced parameters improve compression and may slightly improve convergence speed due to fewer parameters to train.

## Diff from previous best
+1 lines / -1 lines (vs current best)
