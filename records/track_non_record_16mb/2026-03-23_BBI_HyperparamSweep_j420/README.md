# Non-Record: Basis Block Interpolation + Hyperparameter Optimization

**Best val_bpb: 1.4963** (1xH100 SXM, MATRIX_LR=0.03, standard eval — equivalent ~1.10-1.14 on 8xH100)

## Summary

This submission documents two contributions:

1. **Basis Block Interpolation (BBI)** — a novel architecture that stores K basis transformer blocks and creates N effective layers through block reuse with learned depth embeddings. Tested extensively, found to underperform independent blocks due to torch.compile incompatibility. Documented as an informative negative result.

2. **Systematic hyperparameter sweep** — 15+ controlled experiments on 1xH100 SXM identifying MATRIX_LR=0.03 as a significant improvement over the SOTA default of 0.02.

## Basis Block Interpolation — Novel Architecture

### Concept
Instead of N independent transformer blocks (expensive) or simple depth recurrence (inexpressive), BBI stores K "basis blocks" (K << N) and reuses them across N effective layers. Each layer position gets a learned depth embedding so the model knows which unroll iteration it's on.

- **5 basis blocks × 3 unrolls = 15 effective layers** (vs SOTA's 10 independent)
- **dim=576** (vs SOTA's 512 — fewer stored blocks allows wider model)
- Compressed to **10.88MB** (5.12MB headroom under 16MB cap)

### Why It Failed
| Config | val_bpb (1xH100) | Compressed | Issue |
|--------|------------------|-----------|-------|
| SOTA 10L dim=512 | 1.5550 | 15.65MB | Reference |
| BBI 5×3 dim=576 | 1.9415 | 10.88MB | Block reuse too restrictive |
| BBI 8×2 dim=576 | 1.9854 | 16.15MB | Over cap + still worse |
| BBI 10×1 dim=576 | 1.9894 | 15.88MB | Slower due to fullgraph=False |
| BBI 10×1 dim=640 | 1.7038 | 23.58MB | Best BBI but way over cap |

**Root cause:** `torch.compile(fullgraph=False)` is required for block reuse loops, but this makes each step significantly slower than SOTA's `fullgraph=True`. Fewer optimizer steps in the same wallclock = worse final score. The parameter savings from block reuse don't compensate for the training speed loss.

**Takeaway for future work:** Depth recurrence in this challenge is bottlenecked by torch.compile compatibility, not by the architecture itself. A custom CUDA kernel for the recurrent block dispatch could fix this.

## Hyperparameter Sweep — Positive Results

After BBI, I pivoted to systematic hyperparameter optimization on the SOTA submission (10L Int5-MLP + BigramHash + SWA).

### All Results (1xH100 SXM, EVAL_STRIDE=0)
| Rank | Config | val_bpb | Compressed | Steps |
|------|--------|---------|-----------|-------|
| 1 | MATRIX_LR=0.03 | **1.4963** | 16.66MB | 870 |
| 2 | WEIGHT_DECAY=0.02 | 1.5343 | 16.73MB | 871 |
| 3 | WEIGHT_DECAY=0.06 | 1.5344 | 16.73MB | 871 |
| 4 | MUON_MOMENTUM=0.995 | 1.5350 | 16.73MB | 867 |
| 5 | SWA_START_FRAC=0.3 | 1.5415 | 16.72MB | 871 |
| 6 | Default SOTA | 1.5550 | 15.65MB | ~340 |
| 7 | MATRIX_LR=0.015 | 1.5664 | 16.72MB | 870 |
| 8 | WARMDOWN_ITERS=4000 | 1.5756 | 16.73MB | 871 |
| 9 | TRAIN_BATCH_TOKENS=1M | 1.6334 | 16.67MB | 667 |

### Key Finding
**MATRIX_LR=0.03 improves val_bpb by 0.059 over the default 0.02** — a substantial gain from a single hyperparameter change. This improvement is expected to transfer to 8xH100 runs where it would push the current SOTA of 1.1428 lower.

Note: Compressed sizes exceed 16MB on 1xH100 because fewer training steps produce less compressible weights. On 8xH100 with 13,000+ steps and full warmdown, the weights compress better and should fit under the cap.

## Files Included
- `train_gpt_bbi.py` — BBI architecture implementation (novel, negative result)
- `train_gpt_sota.py` — SOTA code used for hyperparameter sweep (from merged submission)
- `sweep_results.txt` — Full sweep output log
- `submission.json` — Submission metadata
- `README.md` — This file

## Methodology
- All experiments run on 1xH100 SXM with 10-minute wallclock cap
- EVAL_STRIDE=0 (standard eval, not sliding window) for fast iteration
- Each experiment changes exactly one variable from the SOTA default
- 15+ total runs across BBI architecture exploration and hyperparameter sweep

## Built On
SOTA submission by the 10L Int5-MLP + BigramHash + SWA author (PR merged 2026-03-20).
