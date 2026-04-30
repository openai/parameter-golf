# PR 1331 — MuonEq-R + 3-Layer Recurrence + WD=0.095 + MLR=0.022 + All-Int6

**Author:** dexhunter
**Claimed BPB:** 1.0900 (3-seed mean, std 0.0005), pre-quant 1.0994
**Artifact size:** ~15.96 MB (max 15,964,018 bytes)
**Seeds:** [42, 0, 7] — scores [1.08980, 1.08953, 1.09053]
**Track:** 10min_16mb
**Hardware:** 8xH100 SXM, 590s train + ~81s eval
**Date:** 2026-04-04
**Base PR:** #1218 (clarkkev); built on #1285

## Files retrieved
- `records__track_10min_16mb__2026-04-04_MuonEqR_3LayerRecurrence_WD095_MLR022_AllInt6__README.md`
- `records__track_10min_16mb__2026-04-04_MuonEqR_3LayerRecurrence_WD095_MLR022_AllInt6__submission.json`
- `records__track_10min_16mb__2026-04-04_MuonEqR_3LayerRecurrence_WD095_MLR022_AllInt6__train_gpt.py`

## Environment variables (from run command)
`NCCL_NET=Socket`, `DATA_DIR=./data`, `SEED=42|0|7`, `MIXED_QUANT=1`, `N_INT6_LAYERS=66`, `RECUR_LAYERS=3,4,5`, `MUON_WD=0.095`, `EMBED_WD=0.095`, `MATRIX_LR=0.022`.

## Claimed changes (from README, verbatim)
"Changes from PR #1285: val_bpb 1.09124 -> 1.08995. Recurrence Layers 4,5 (2-layer) -> Layers 3,4,5 (3-layer). Weight decay 0.090 -> 0.095. Matrix LR 0.020 -> 0.022. Everything else same. Key Innovations: (1) 3-Layer Depth Recurrence — Layers 3, 4, and 5 are repeated (RECUR_LAYERS=3,4,5), creating 14 virtual layers from 11 physical. MLP weights fully shared. ~0.0005 BPP improvement over 2-layer recurrence. (2) WD=0.095 + MLR=0.022 Synergy — Higher weight decay compresses weights better, slightly higher matrix LR recovers quality. The 3-layer recurrence needs WD>=0.093 to fit all-int6 under 16MB. (3) MuonEq-R + All-Int6 GPTQ — Row-normalized Muon optimizer with all 66 layers at int6 precision (carried from PR #1285). No TTT, no SLOT."
