# PR 1415 — SP4096 + 3-Layer Recurrence + GPTQ Embeddings + SDClip + ETLB

**Author:** bigbag
**Claimed BPB:** 1.0913 (3-seed mean, std 0.0012)
**Artifact size:** ~14,746,381 bytes mean
**Seeds:** 42, 314, 999

## Files retrieved
- `records__track_10min_16mb__2026-04-06_SP4096_3LayerRecur_GPTQ-Embed_SDClip_ETLB__README.md`
- `records__track_10min_16mb__2026-04-06_SP4096_3LayerRecur_GPTQ-Embed_SDClip_ETLB__train_gpt.py`
- `records__track_10min_16mb__2026-04-06_SP4096_3LayerRecur_GPTQ-Embed_SDClip_ETLB__submission.json`

## Environment variables (from run command)
VOCAB_SIZE=4096, QK_GAIN_INIT=5.0, MUON_WD=0.095, MATRIX_LR=0.022, LOOP_START=3, LOOP_END=5, ETLB_ENABLED=1, ETLB_LR=0.05, ETLB_STEPS=5, ETLB_CLIP=3.0, SEED=42

## Claimed changes (from README, verbatim)
> Key Techniques:
> 1. SP4096 Vocabulary — 4096-token SentencePiece BPE
> 2. GPTQ on Embeddings (int8) — PR #1394 innovation, saves ~2MB artifact vs FP16 embeddings
> 3. SDClip — std-dev based quantization clip thresholds (PR #1394)
> 4. 3-Layer Depth Recurrence (layers 3,4,5, from step ~2950) — extends PR #1204
> 5. QK-Gain 5.0 — PR #1217
> 6. WD=0.095 + MLR=0.022 — higher WD for compression, higher LR to compensate (PR #1331)
> 7. ETLB: Eval-Time Logit Bias — optimizes vocab bias vector during sliding window eval (PR #1399)
> 8. MuonEq-R — row-normalize before Newton-Schulz (PR #1260)
> 9. Full GPTQ int6 + Brotli compression
> 10. LZMA code wrapper — minified code saves ~40KB artifact
>
> Compliance: No SLOT, no TTT (no model weight updates during eval), ETLB only modifies a separate bias vector, standard autoregressive sliding-window eval (stride=64).
