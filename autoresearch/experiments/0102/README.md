# Experiment 102

**Date:** 2026-03-19T15:52:10.294381+00:00
**Result:** KEPT
**val_bpb:** 1.5796
**Artifact size:** 15,367,172 bytes
**Model params:** 28863584
**Last step:** 319
**Pre-quant val_bpb:** 1.5639
**Quantization gap:** 0.0157
**Propose time:** 175.0s
**Train time:** 283.6s

## Change
Increased matrix_lr (Muon optimizer LR for weight matrices) from 0.05 to 0.06 (+20%). Experiment #3 showed that 0.08 improved BPB from 1.5956 to 1.5698 but exceeded the 16MB artifact cap (16.6MB). A moderate step to 0.06 should capture some of that convergence improvement while keeping compressed size well within budget (~15.4MB estimated). Muon's Newton-Schulz orthogonalization bounds update magnitudes, making it robust to moderate LR increases.

## Diff from previous best
Identical to current best
