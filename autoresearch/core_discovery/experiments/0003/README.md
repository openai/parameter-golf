# Experiment 3

**Date:** 2026-03-19T17:55:21.865584+00:00
**Lane/Stage:** core/discovery
**Result:** KEPT
**val_bpb:** 1.5574
**Artifact size:** 9,901,249 bytes
**Model params:** 17059912
**Last step:** 366
**Pre-quant val_bpb:** 1.5457
**Quantization gap:** 0.0117
**Eval time:** 11071 ms
**Peak memory:** 10239 MiB
**Gate reason:** improved_val_bpb (none -> 1.5574)
**Propose time:** 37.4s
**Train time:** 305.8s

## Change
Increase matrix_lr from 0.04 to 0.06 to improve short-horizon convergence speed. Higher matrix_lr (Muon optimizer LR for weight matrices) has shown consistent BPB improvements in prior experiments (0.05 and 0.08 both helped). The 9x512 model has ample size headroom (~15.8MB vs 16MB cap), so this change carries no size risk and should not affect step time.

## Diff from previous best
Identical to current best
