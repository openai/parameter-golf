# Experiment 054: relu² 3x + OrthoInit + SmearGate + BigramHash

## Status: COMPLETED

## Config
- relu² 3x (h=1536), plain Muon, int6 QAT
- OrthoInit + SmearGate + BigramHash (PR135 features)
- TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216, GRAD_CLIP_NORM=0.3
- No SWA, no LAWA

## Results
| Metric | Value |
|--------|-------|
| Steps | 10,633 @ 56.5ms/step |
| Artifact | 16,757,760 bytes ❌ (758KB over — BigramHash adds params) |
| **Standard eval** | **1.1738 BPB** |
| **Sliding eval s64** | **1.1622 BPB** ← NEW BEST |

## Progression
| Exp | Change | Sliding BPB | Diff |
|-----|--------|-------------|------|
| 049 | SwiGLU + NorMuon | 1.1685 | baseline |
| 053 | Switch to relu² 3x | 1.1639 | -0.0046 |
| **054** | **+ OrthoInit + SmearGate + BigramHash** | **1.1622** | **-0.0017** |

## Gap to PR135 (1.1539): 0.0083
Remaining differences: seq2048+batch786K, no QAT, possibly tuning

## wandb
- Run name: 054_relu2_ortho_smeargate_bigram
