# Experiment 108

**Date:** 2026-03-19T16:34:44.616551+00:00
**Result:** REVERTED
**val_bpb:** 1.5953
**Artifact size:** 15,706,072 bytes
**Model params:** 28863584
**Last step:** 319
**Pre-quant val_bpb:** 1.5809
**Quantization gap:** 0.0144
**Propose time:** 299.3s
**Train time:** 280.6s

## Change
Increased tied_embed_lr from 0.05 to 0.08 (+60%). The tied embedding (1024×512) is the shared input/output interface of the model, critical for both token encoding and prediction. This LR hasn't been tuned since scaling to 12 layers. Following the pattern of successful Muon LR increases (matrix_lr 0.05→0.07), a moderate Adam LR increase for the embedding should improve convergence on the short-horizon proxy where warmdown suppresses effective LR to ~30%. Zero impact on artifact size or step time.

## Diff from previous best
+1 lines / -1 lines (vs current best)
