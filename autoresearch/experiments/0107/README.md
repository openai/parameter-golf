# Experiment 107

**Date:** 2026-03-19T16:25:38.396170+00:00
**Result:** KEPT
**val_bpb:** 1.5617
**Artifact size:** 15,894,735 bytes
**Model params:** 28863584
**Last step:** 325
**Pre-quant val_bpb:** 1.5475
**Quantization gap:** 0.0142
**Propose time:** 0.0s
**Train time:** 282.8s

## Change
Increased matrix_lr (Muon optimizer LR for weight matrices) from 0.06 to 0.07 (+17%). Experiments show 0.06→1.5796 BPB at 15.4MB (#102) and 0.08→1.5698 BPB at 16.6MB (#3, over cap). Linear interpolation predicts 0.07 should yield ~1.5747 BPB at ~15.97MB, capturing half the BPB improvement while staying just under the 16MB artifact cap. Muon's Newton-Schulz orthogonalization bounds update magnitudes, making it robust to moderate LR increases.

## Diff from previous best
Identical to current best
