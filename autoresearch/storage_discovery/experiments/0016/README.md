# Experiment 16

**Date:** 2026-03-19T19:12:02.557902+00:00
**Lane/Stage:** storage/discovery
**Result:** REVERTED
**val_bpb:** 1.6628
**Artifact size:** 9,114,422 bytes
**Model params:** 17059912
**Last step:** 329
**Pre-quant val_bpb:** 1.6281
**Quantization gap:** 0.0347
**Eval time:** 11009 ms
**Peak memory:** 10240 MiB
**Gate reason:** no_storage_improvement
**Propose time:** 279.5s
**Train time:** 280.0s

## Change
Keep the tied embedding (tok_emb.weight) as fp16 passthrough instead of int8 quantization. This tensor is uniquely critical because it's used for both input token lookup AND output logit projection (tied embeddings), so int8 quantization error is amplified twice — at the first and last steps of every forward pass. Storing it as fp16 costs only ~500KB extra (524K params × 2 bytes vs 1 byte = ~500KB increase) which is well within the 16MB budget (current artifact ~8.8MB), and should reduce the quantization gap by eliminating quantization noise from the model's most dual-use tensor.

## Diff from previous best
+51 lines / -26 lines (vs current best)
