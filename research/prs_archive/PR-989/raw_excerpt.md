# PR 989 — QAT x SWA Ablation: Antagonistic Interaction in Quantization-Aware Training

**Author:** not stated in README (non-record research submission)
**Claimed BPB:** 1.14018 (no_swa_qat, 3-seed mean, std ±0.00044); delta -3.64 mBPB vs control (1.14382)
**Artifact size:** ~15.8-16.4 MB depending on config; 10% magnitude pruning for QAT configs, 5% for non-QAT
**Seeds:** 42, 1337, 2024

## Files retrieved
- `records__track_10min_16mb__2026-03-28_QAT_SWA_Ablation__README.md`
- `records__track_10min_16mb__2026-03-28_QAT_SWA_Ablation__submission.json`
- `records__track_10min_16mb__2026-03-28_QAT_SWA_Ablation__train_gpt.py`
- `records__track_10min_16mb__2026-03-28_QAT_SWA_Ablation__run.sh`
- `records__track_10min_16mb__2026-03-28_QAT_SWA_Ablation__run_matrix.sh`

## Environment variables (from run command in README)
bash run.sh no_swa_qat 42 ; bash run_matrix.sh (full 2x2 matrix)

## Claimed changes (from README, verbatim)
> This is a non-record research submission. We present a systematic 2x2 factorial ablation of QAT x SWA interaction, revealing that SWA and QAT are antagonistic mechanisms.

> Full 2x2 Factorial (2-seed means):
> no_swa_qat (QAT=Yes, SWA=No): 1.14018 — -3.64 mBPB vs control (1st)
> control (QAT=No, SWA=Yes): 1.14382 — baseline (2nd)
> qat_snap70 (QAT=Yes, SWA=Yes): 1.14468 — +0.86 mBPB (3rd)
> no_swa (QAT=No, SWA=No): 1.14486 — +1.04 mBPB (4th)

> QAT without SWA wins. SWA + QAT interfere: when both enabled, the result is worse than either alone. QAT is 3.5x stronger than SWA (3.64 vs 1.04 mBPB).

> Why SWA and QAT Conflict: SWA averages checkpoints producing smooth weight distributions that quantize well passively. QAT uses Straight-Through Estimator (STE) fake-quantization during training, actively shaping weights for quantization boundaries. When combined, SWA's averaging dilutes QAT's quantization-aware adjustments.

> Architecture: Based on PR #180 stack (10L/512d/MLP3x). Quantization int5 MLP / int6 attention + zstd-22. QAT start: 70% of training (snap at step ~4550), int6 per-row STE.

> Training val_bpb is misleading for QAT: QAT shows worse training metrics (1.1623 vs 1.1538) but better post-quantization BPB.

> Known limitation: Based on older stack (no EMA, XSA, Partial RoPE). Top entries (#549 at 1.1194) use EMA instead of SWA — EMA x QAT interaction is an open question.
