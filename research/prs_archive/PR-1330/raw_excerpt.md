# PR 1330 — JEPA v2: Why Single-Step JEPA Collapses, and How to Fix It

**Author:** luciobaiocchi
**Claimed BPB:** 1.4617 pre-quant (bigram mode); JEPA mode 1.6224; roundtrip int6+lzma 1.7594 (bigram)
**Artifact size:** 8.52-8.70 MB (non-JEPA), 5.53-5.64 MB (JEPA modes)
**Seeds:** not stated (single run per mode)
**Track:** non_record_unlimited_compute
**Hardware:** RTX 5060 Ti 16GB
**Date:** 2026-04-04

## Files retrieved
- `records__track_non_record_16mb__2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA__README.md`
- `records__track_non_record_16mb__2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA__agent__changes.md`
- `records__track_non_record_16mb__2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA__run.sh`
- `records__track_non_record_16mb__2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA__submission.json`
- `records__track_non_record_16mb__2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA__train_gpt.py`

## Environment variables (from run script)
`USE_JEPA`, `JEPA_LAMBDA=0.12`, `JEPA_EMA_MOMENTUM=0.9`, `BIGRAM_VOCAB_SIZE=2048`, `MLP_LEAKY_SLOPE=0.5`, `MAX_WALLCLOCK_SECONDS=600/1800/120`, `TRAIN_BATCH_TOKENS=131072/65536`, `VAL_LOSS_EVERY`, `TRAIN_LOG_EVERY`, `ARTIFACT_EMA_DECAY=0.99`, `QUANT_MAX=31`, `TOKENIZER_PATH`, `DATA_PATH`.

## Claimed changes (from README, verbatim excerpt)
"JEPA v2: diagnoses why same-sequence next-k JEPA collapses in causal LMs... Bug 1: EMA Momentum Too High -> Task Trivially Easy (fix JEPA_EMA_MOMENTUM=0.9). Bug 2: Single-Step Prediction Is Redundant With CE (fix: multi-step offsets [1,2,4,8] weights [1.0, 0.5, 0.25, 0.125]). Bug 3: Gradient Accumulation Batch Mismatch (fix: target encoder runs inside micro-step loop). Additional: int6 quantization (range [-31,31]), LZMA compression (preset=9), BigramHash embedding (Cantor pairing, 2048 vocab dim 512), Artifact EMA decay=0.9999, LeakyReLU(0.5)^2. Architecture: 11L 512-dim U-Net Transformer, 5 encoder + 6 decoder skip connections, mlp_mult=3, GQA 8q/4kv. Key finding: JEPA collapse persists after fixing all three bugs; geometric root cause — consecutive positions share almost all context."
