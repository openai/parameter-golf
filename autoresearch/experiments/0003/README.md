# Experiment 3

**Date:** 2026-03-19T15:15:40.593860+00:00
**Result:** REVERTED
**val_bpb:** 1.5698
**Artifact size:** 16,575,250 bytes
**Propose time:** 279.6s
**Train time:** 324.7s

## Change
Increased matrix_lr (Muon optimizer learning rate for weight matrices) from 0.05 to 0.08 (+60%). With limited training time (180s on 1 GPU, ~few hundred steps), the model needs faster convergence per step. Muon's Newton-Schulz orthogonalization inherently bounds update magnitudes, making it robust to higher LRs. The current 0.05 was tuned for longer training regimes; a higher rate should better exploit the limited training budget by learning more per step.

## Diff from previous best
+7 lines / -6 lines (vs current best)
