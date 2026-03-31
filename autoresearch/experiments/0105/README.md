# Experiment 105

**Date:** 2026-03-19T16:13:56.045161+00:00
**Result:** REVERTED
**val_bpb:** 1.5456
**Artifact size:** 21,588,938 bytes
**Model params:** 28863584
**Last step:** 307
**Pre-quant val_bpb:** 1.5441
**Quantization gap:** 0.0015
**Propose time:** 0.0s
**Train time:** 295.7s

## Change
Changed LR warmdown from step-count-based (warmdown_iters*step_ms) to time-fraction-based (warmdown_frac*total_time). The old formula was catastrophically broken on the 1×H100 proxy: with warmdown_iters=1200 and ~500ms/step, the warmdown window (600s) exceeded total training time (180s), causing the LR to be scaled to ~25% from step 1 onward. The new warmdown_frac=0.085 gives identical behavior on 8×H100 (51s warmdown) while correctly using only the last 15.3s (~8.5%) for warmdown on the proxy, letting the model train at full learning rate for ~92% of the run.

## Diff from previous best
+16 lines / -7 lines (vs current best)
