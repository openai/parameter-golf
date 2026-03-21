# Experiment 048: PR122 Merged + SwiGLU (small model baseline)

## Status: COMPLETED

## Config
- Our merged script (train_gpt_merged.py) with NorMuon + int6 QAT + bit-packing
- USE_SWIGLU=1, SWIGLU_HIDDEN=668, TRAIN_SEQ_LEN=1024
- SWA_ENABLED=1 (default), LAWA_ENABLED=0
- 9 layers, dim=512, vocab=1024
- Train data, 8xH100 NV18 NVLink

## Results
| Metric | Value |
|--------|-------|
| Steps | 11,986 @ 50.0ms/step |
| Artifact | 12,735,738 bytes ✅ (3.2MB headroom!) |
| Standard eval | **1.2325 BPB** |
| Sliding eval stride=64 | **1.1990 BPB** |
| SWA | 9 checkpoints averaged |

## Key Findings
- Int6 bit-packing saved massive space (12.7MB vs 15.95MB) — 3.2MB freed
- But BPB worse than 038 (1.1935) — NorMuon + SWA may be hurting vs plain Muon
- The headroom enabled experiment 049 with h=1024

## wandb
- Run ID: di7q4oni
- Run name: 048_pr122_merged_swiglu
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/di7q4oni
