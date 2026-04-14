# PR 1408 — dTTT + BigramHash 3072×112

**Author:** Aamod Bhatt (aamodbhatt) — per submission.json (user context notes abaybektursun)
**Claimed BPB:** 1.0800 (3-seed mean, std 0.0002)
**Artifact size:** ~15,873,363–15,895,227 bytes
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_10min_16mb__2026-04-06_dTTT_BH3072_11L_8xH100__README.md`
- `records__track_10min_16mb__2026-04-06_dTTT_BH3072_11L_8xH100__train_gpt.py`
- `records__track_10min_16mb__2026-04-06_dTTT_BH3072_11L_8xH100__submission.json`

## Environment variables (from run command)
BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=112, ETLB_ENABLED=0, TTT_ENABLED=1, TTT_LR=0.0005, TTT_EPOCHS=10, TTT_BATCH_SEQS=32, TTT_FREEZE_BLOCKS=0, TTT_GRAD_CLIP=1.0, TTT_COSINE_DECAY=1, XSA_LAST_N=11, EMA_ENABLED=0, SWA_ENABLED=1, SWA_EVERY=50, ROPE_DIMS=16, LN_SCALE=1, VE_ENABLED=1, VE_DIM=128, VE_LAYERS=9,10, MUON_WD=0.04, ADAM_WD=0.04, MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035, MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500, WARMDOWN_ITERS=4000, ITERATIONS=20000, MAX_WALLCLOCK_SECONDS=600, EVAL_STRIDE=64, QK_GAIN_INIT=5.0, SEED=1337

## Claimed changes (from README, verbatim)
> Builds directly on PR #1351 (Discriminative TTT) with one modification:
>
> 1. BigramHash 3072×112 (up from 2048×128 in PR #1351). More expressive n-gram context features — 3072×112 follows PR #1019 and PR #1405 best practices for the current architecture.
>
> All other hyperparameters identical to PR #1351: dTTT 10 epochs, AdamW LR=0.0005, freeze=0, per-block LR scaling 0.3×→1.0×, cosine decay, GPTQ int6 damp=0.005, QK_GAIN=5.0, WARMDOWN=4000, XSA all-layers, ROPE_DIMS=16.
>
> Compliance (Track A — Fixed Predictor): No eval-time adaptation of any kind. Score-first ordering: standard autoregressive sliding-window eval. No n-gram cache or external data at eval. Single left-to-right pass.
