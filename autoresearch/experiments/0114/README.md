# Experiment 114

**Date:** 2026-03-19T17:43:37.445039+00:00
**Result:** REVERTED
**val_bpb:** 1.5696
**Artifact size:** 15,706,554 bytes
**Model params:** 28863584
**Last step:** 320
**Pre-quant val_bpb:** 1.5557
**Quantization gap:** 0.0139
**Propose time:** 511.9s
**Train time:** 282.0s

## Change
Reduced Muon optimizer target momentum from 0.95 to 0.92. Experiment #111 showed that reaching high momentum faster (warmup_steps 500→100) hurt performance (1.5710 vs 1.5617), suggesting momentum=0.95 is too high for the short-horizon proxy where only ~300 steps are available. Lower momentum gives more responsive gradient signal to the Newton-Schulz orthogonalization, reducing reliance on stale gradient history. On the 1xH100 proxy, the warmup (0.85→0.92 over 500 steps) reaches ~0.892 at step 300 (vs ~0.910 currently), giving meaningfully fresher gradients. Zero impact on parameter count, step time, or artifact size.

## Diff from previous best
+1 lines / -1 lines (vs current best)
