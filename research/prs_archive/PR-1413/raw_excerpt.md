# PR 1413 — SP8192 + QK-Gain 5 + Legal Score-First TTT

**Author:** dexhunter (note: user context lists clarkkev; submission.json says dexhunter)
**Claimed BPB:** 1.08279 (3-seed mean, std ~0.00049)
**Artifact size:** ~15,990,874 bytes mean
**Seeds:** 0, 42, 1234

## Files retrieved
- `records__track_10min_16mb__2026-04-06_SP8192_QK5_LegalTTT_1.0828__README.md`
- `records__track_10min_16mb__2026-04-06_SP8192_QK5_LegalTTT_1.0828__train_gpt.py`
- `records__track_10min_16mb__2026-04-06_SP8192_QK5_LegalTTT_1.0828__submission.json`

## Environment variables (from run command)
NCCL_NET=Socket, QK_GAIN_INIT=5.0, TTT_ENABLED=1, TTT_LR=0.005, TTT_EPOCHS=3, SEED={0,42,1234}

## Claimed changes (from README, verbatim)
> A single-knob improvement on top of @clarkkev's PR #1394 sp8192 baseline + a legal score-first TTT eval pass:
>
> 1. QK_GAIN_INIT = 5.0 (vs PR #1394's 4.0)
> 2. Legal score-first TTT — score each sliding window chunk under inference_mode() BEFORE any gradient update; each chunk is only trained on after it has been fully scored.
>
> Strict score-before-update ordering matches PR #549 precedent and satisfies Issue #1017 conditions 1–4. No eval-time delta optimization (no SLOT), no pre-quant TTT on val data, no two-pass rescoring, no n-gram cache.
>
> Changes from baseline PR #1394: QK_GAIN_INIT 4.0→5.0; TTT added (Legal score-first, LR=0.005, epochs=3, freeze=0). All other components unchanged: SentencePiece BPE 8192, 11L/512d/8H/4KV, MLP 4x, Partial RoPE 16d, depth recurrence loop layers 4-5 twice, MuonEq-R WD=0.085, GPTQ int6 matrices + int8 embeddings + SD-clip.
