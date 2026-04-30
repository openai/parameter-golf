# PR 1217 — Record: MuonEq-R + Context-Only SLOT + QK_GAIN=5.0

**Author:** Pavel Liashkov (bigbag)
**Branch date:** 2026-04-01
**Claimed BPB:** 1.1027 (3-seed mean, std 0.0011) — context-SLOT; TTT variant = 1.11107
**Artifact size:** ~15.80 MB (mean 15,795,153)
**Seeds:** 1337, 42, 2024
**Hardware:** 8×H100 SXM, ~88.8 ms/step, ~6654 steps

## Files retrieved
- `records__track_10min_16mb__2026-04-01_MuonEqR_ContextSLOT_QKGain5__README.md`
- `records__track_10min_16mb__2026-04-01_MuonEqR_ContextSLOT_QKGain5__train_gpt.py`
- `records__track_10min_16mb__2026-04-01_MuonEqR_ContextSLOT_QKGain5__submission.json`

## Environment variables (from reproduction command)
`QK_GAIN_INIT=5.0 SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 SEED=$SEED`

## Claimed changes (from README, verbatim)
"Built on PR #1179 (@dexhunter) with three additions:

- MuonEq-R (row-normalization before Newton-Schulz) -- from arXiv:2603.28254
- QK_GAIN_INIT=5.0 -- our hyperparameter sweep (monotonic gains from 1.5 to 5.0)
- Context-Only SLOT -- causal variant of SLOT that optimizes delta using only already-scored context tokens

Training: ~600s. Eval (sliding + context-only SLOT): ~190s. Total: ~13 min end-to-end.

Beats merged SOTA (PR #1019, 1.1147) by 0.012 BPB (p << 0.01)."
