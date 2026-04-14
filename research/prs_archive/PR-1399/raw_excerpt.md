# PR 1399 — Pre-Quant TTT + ETLB: Eval-Time Logit Bias

**Author:** AnubhavBharadwaaj
**Claimed BPB:** 1.0898 (3-seed mean, std 0.0008)
**Artifact size:** ~16,084,685–16,092,287 bytes
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_10min_16mb__2026-04-05_PreQuantTTT_ETLB_MuonEqR_DepthRecurrence_AllInt6__README.md`
- `records__track_10min_16mb__2026-04-05_PreQuantTTT_ETLB_MuonEqR_DepthRecurrence_AllInt6__train_gpt.py`
- `records__track_10min_16mb__2026-04-05_PreQuantTTT_ETLB_MuonEqR_DepthRecurrence_AllInt6__submission.json`

## Environment variables (from run command)
SEED, PRE_QUANT_TTT=1, PRE_QUANT_TTT_LR=0.0005, PRE_QUANT_TTT_EPOCHS=1, PRE_QUANT_TTT_FREEZE=9, MUON_WD=0.090, EMBED_WD=0.090, QK_GAIN_INIT=5.0, ETLB_ENABLED=1, ETLB_LR=0.05, ETLB_STEPS=5, ETLB_CLIP=3.0

## Claimed changes (from README, verbatim)
> This submission introduces **Eval-Time Logit Bias (ETLB)**, a novel eval-time augmentation technique that optimizes a warm-started vocabulary bias vector during sliding window evaluation. Combined with pre-quantization test-time training (Pre-Quant TTT), this achieves a new best pure neural BPB on the 10-minute 16MB track. Built on PR #1285's architecture (MuonEq-R + Depth Recurrence + All-Int6 GPTQ).
>
> Pre-Quant TTT: Adapts the full-precision EMA model weights on validation data before GPTQ quantization. Freeze first 9 of 11 blocks, AdamW lr=0.0005, validation chunks 32768 tokens, 1 epoch. Score-first compliant.
>
> ETLB: Optimizes a bias vector b ∈ ℝ^vocab added to output logits. SGD 5 steps lr=0.05, clip [-3,3], warm-started across windows. Operates purely in logit space, no model weight modification. Consistent ~0.002 BPB improvement.
