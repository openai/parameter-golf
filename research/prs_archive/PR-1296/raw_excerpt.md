# PR 1296 — SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + QK-Gain 5.0

**Author:** aryanbhosale
**Claimed BPB:** 1.08971631 (3-seed mean, std 0.00028794; seeds 42, 314, 999; per-seed 1.08938974, 1.08982552, 1.08993367)
**Artifact size:** ~15.99 MB (per-seed: 15,999,165 / 15,997,318 / 15,990,607 bytes)
**Seeds:** 42, 314, 999
**Track:** 10min_16mb
**Delta vs PR #1019:** -0.02502 BPB

## Files retrieved
- `records__track_10min_16mb__2026-04-03_SP4096_DepthRecurrence_MuonEqR_GPTQ__README.md`
- `records__track_10min_16mb__2026-04-03_SP4096_DepthRecurrence_MuonEqR_GPTQ__submission.json`
- `records__track_10min_16mb__2026-04-03_SP4096_DepthRecurrence_MuonEqR_GPTQ__train_gpt.py`

## Environment variables (from README run command)
SEED=42, RECUR_LAYERS=4,5, RECUR_START_STEP=3000, PARALLEL_START_LAYER=7

## Claimed changes (from README, verbatim)

> Key Techniques: 1. 4096-Vocab + MLP 4x + WD 0.090 — PR #1218 @clarkkev, PR #1285 @dexhunter. 2. Depth Recurrence (layers 4,5) — PR #1204 @msisovic, PR #1260 @dexhunter. 3. Parallel Residuals (from layer 7) — PR #1204 @msisovic, PR #1289 @MatoTeziTanka. 4. MuonEq-R — arXiv:2603.28254, PR #1260 @dexhunter. 5. QK-Gain 5.0 — PR #1217 @bigbag. 6. Full GPTQ int6 + Brotli + Compressed Wrapper.

> Compliance: No TTT, no SLOT, no n-gram cache, no eval-time adaptation.

> 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128). Merged SOTA (PR #1019): 1.1147 BPB. Delta: -0.0250 BPB.
