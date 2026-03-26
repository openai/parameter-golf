# Experiment 109

**Date:** 2026-03-19T16:44:24.645567+00:00
**Result:** REVERTED
**val_bpb:** 1.5846
**Artifact size:** 16,084,029 bytes
**Model params:** 28863584
**Last step:** 327
**Pre-quant val_bpb:** 1.5723
**Quantization gap:** 0.0123
**Propose time:** 0.0s
**Train time:** 282.3s

## Change
Increased tied_embed_lr from 0.05 to 0.08 (+60%). The tied embedding table (1024×512) serves as both input embedding and output projection, making it critical for both token representation and prediction quality. This LR hasn't been tuned yet, while the analogous matrix_lr increase (0.05→0.06→0.07) yielded consistent BPB improvements. With limited training steps (~300 on proxy), faster embedding convergence should improve both input representations and output predictions. The embedding has only 524K params (1.8% of model), is optimized with Adam (robust to LR changes), and on the 1xH100 proxy the effective LR is further suppressed by the warmdown schedule (~30% of base), making 0.08 a conservative step. Zero impact on parameter count, artifact size, or step time.

## Diff from previous best
+1 lines / -1 lines (vs current best)
