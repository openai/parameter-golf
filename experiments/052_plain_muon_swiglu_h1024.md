# Experiment 052: Plain Muon + SwiGLU h=1024 + seq4096 + grad_clip=0.3

## Status: COMPLETED

## Config
- USE_SWIGLU=1, SWIGLU_HIDDEN=1024, TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216
- USE_NORMUON=0 (plain Muon), GRAD_CLIP_NORM=0.3
- int6 QAT, no SWA, no LAWA

## Results
| Metric | Value |
|--------|-------|
| Steps | 10,429 @ 57.5ms/step |
| Artifact | 16,214,612 bytes ❌ |
| Standard eval | 1.1819 BPB |
| Sliding s64 | **1.1701 BPB** |

## Comparison with 049 (NorMuon)
| | 049 (NorMuon) | 052 (Muon+clip) | Diff |
|--|--------------|-----------------|------|
| Sliding | **1.1685** | 1.1701 | +0.0016 (NorMuon better) |
| Standard | **1.1805** | 1.1819 | +0.0014 |

## Key Finding
Plain Muon + grad_clip=0.3 did NOT help for SwiGLU. NorMuon is actually slightly better.
This is opposite to what the relu² PRs show — NorMuon may specifically benefit SwiGLU.
grad_clip=0.3 may be too aggressive for this config.

## wandb
- Run ID: check dashboard
- Run name: 052_plain_muon_swiglu_h1024
