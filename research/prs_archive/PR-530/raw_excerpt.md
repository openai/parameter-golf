# PR 530 — Non-Record: Basis Block Interpolation + Hyperparameter Optimization

**Author:** j420
**Claimed BPB:** val_bpb 1.4963 (1xH100 SXM, MATRIX_LR=0.03, standard eval / not sliding window). Author notes: "Equivalent ~1.10-1.14 on 8xH100."
**Artifact size:** 16.66 MB (best config; exceeds 16 MB cap on 1xH100 per README note)
**Seeds:** 42

## Files retrieved
- `records__track_non_record_16mb__2026-03-23_BBI_HyperparamSweep_j420__README.md`
- `records__track_non_record_16mb__2026-03-23_BBI_HyperparamSweep_j420__submission.json`
- `records__track_non_record_16mb__2026-03-23_BBI_HyperparamSweep_j420__train_gpt_bbi.py`
- `records__track_non_record_16mb__2026-03-23_BBI_HyperparamSweep_j420__train_gpt_sota.py`

## Claimed changes (from README, verbatim)

> This submission documents two contributions:
>
> 1. Basis Block Interpolation (BBI) — a novel architecture that stores K basis transformer blocks and creates N effective layers through block reuse with learned depth embeddings. Tested extensively, found to underperform independent blocks due to torch.compile incompatibility. Documented as an informative negative result.
>
> 2. Systematic hyperparameter sweep — 15+ controlled experiments on 1xH100 SXM identifying MATRIX_LR=0.03 as a significant improvement over the SOTA default of 0.02.
>
> BBI Concept: Instead of N independent transformer blocks (expensive) or simple depth recurrence (inexpressive), BBI stores K "basis blocks" (K << N) and reuses them across N effective layers. Each layer position gets a learned depth embedding so the model knows which unroll iteration it's on. 5 basis blocks × 3 unrolls = 15 effective layers (vs SOTA's 10 independent); dim=576 (vs SOTA's 512); compressed to 10.88MB (5.12MB headroom).
>
> Why It Failed: BBI 5×3 dim=576 → 1.9415 / 10.88MB. BBI 8×2 dim=576 → 1.9854 / 16.15MB. BBI 10×1 dim=576 → 1.9894 / 15.88MB. BBI 10×1 dim=640 → 1.7038 / 23.58MB. SOTA 10L dim=512 → 1.5550 / 15.65MB. Root cause: torch.compile(fullgraph=False) is required for block reuse loops, but this makes each step significantly slower than SOTA's fullgraph=True. Fewer optimizer steps in the same wallclock = worse final score.
>
> Hyperparameter Sweep (1xH100 SXM, EVAL_STRIDE=0): MATRIX_LR=0.03 → 1.4963 (870 steps). WEIGHT_DECAY=0.02 → 1.5343. WEIGHT_DECAY=0.06 → 1.5344. MUON_MOMENTUM=0.995 → 1.5350. SWA_START_FRAC=0.3 → 1.5415. Default SOTA → 1.5550. MATRIX_LR=0.015 → 1.5664. WARMDOWN_ITERS=4000 → 1.5756. TRAIN_BATCH_TOKENS=1M → 1.6334.
>
> Key Finding: MATRIX_LR=0.03 improves val_bpb by 0.059 over the default 0.02 — a substantial gain from a single hyperparameter change. Expected to transfer to 8xH100 runs.
>
> Methodology: All experiments on 1xH100 SXM with 10-minute wallclock cap. EVAL_STRIDE=0 (standard eval, not sliding window) for fast iteration. Each experiment changes exactly one variable from the SOTA default. 15+ total runs.
