# Experiment 111

**Date:** 2026-03-19T17:12:14.564837+00:00
**Result:** REVERTED
**val_bpb:** 1.5710
**Artifact size:** 16,647,902 bytes
**Model params:** 28863584
**Last step:** 300
**Pre-quant val_bpb:** 1.5629
**Quantization gap:** 0.0081
**Propose time:** 370.7s
**Train time:** 290.8s

## Change
Reduced muon_momentum_warmup_steps from 500 to 100. On the 1xH100 proxy with ~300 training steps, the warmup never completes (momentum reaches only ~0.91 instead of target 0.95). By shortening to 100 steps, the Muon optimizer reaches its target momentum of 0.95 by step 100 (one-third of training), providing better gradient smoothing and acceleration for the remaining 200 steps. Higher momentum gives more effective gradient accumulation, especially valuable when the effective learning rate is already heavily suppressed by the broken warmdown schedule (~30% of nominal). Zero impact on parameter count, step time, or artifact size. On 8xH100, the change is negligible (100 vs 500 out of ~13K steps).

## Diff from previous best
+1 lines / -1 lines (vs current best)
