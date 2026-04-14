# PR 1156 — Record: EGGROLL v2 (Post-GPTQ Bin Refinement)

**Author:** koltondrake
**Branch date:** 2026-03-30
**Claimed BPB:** 1.1161 (3-seed mean, std 0.0001)
**Artifact size:** ~15.3 MB
**Seeds:** 42=1.1163, 1337=1.1160, 2024=1.1161
**Hardware:** 8×H100 SXM, Legal TTT

## Files retrieved
- `records__track_10min_16mb__2026-03-30_EGGROLL_PostGPTQ_BinRefinement__train_gpt.py`
- `records__track_10min_16mb__2026-03-30_EGGROLL_PostGPTQ_BinRefinement__submission.json`

(No README.md in record directory.)

## Claimed changes (from commit message, verbatim)
"val_bpb: 1.1161 | val_loss: 1.884 nats | ~15.3 MB | 8×H100 SXM | Legal TTT

Seeds: 42=1.1163, 1337=1.1160, 2024=1.1161 | Mean=1.1161, Std=0.0001

Novel: EGGROLL Antithetic Ternary Bin Search — post-GPTQ bin refinement
Also: adds missing TTT call to PR #1130 eval pipeline

Built on PR #1130 by @Gusanidas, PR #549 by @abaybektursun"
