# Experiment 055: relu² 3x + PR135 features + seq2048 + batch786K + NO QAT

## Status: COMPLETED

## Config
- relu² 3x, OrthoInit, SmearGate, BigramHash, plain Muon, grad_clip=0.3
- TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=786432, QAT_ENABLED=0
- No SWA, no LAWA

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,372 @ 81.6ms/step |
| Artifact | 16,790,949 bytes ❌ (791KB over) |
| Standard eval | 1.1825 BPB |
| **Sliding eval s64** | **1.1619 BPB** |
| Sliding improvement | 0.0207 BPB (bigger than seq4096's 0.0117) |

## Comparison: seq2048 vs seq4096
| | 054 (seq4096) | 055 (seq2048) |
|--|--------------|--------------|
| Steps | 10,633 | 7,372 |
| ms/step | 56.5 | 81.6 |
| Standard | **1.1738** | 1.1825 |
| Sliding | 1.1622 | **1.1619** |
| Sliding boost | 0.0117 | **0.0207** |

## Key Finding
seq2048 matches seq4096 on sliding eval (1.1619 ≈ 1.1622) despite fewer steps.
PR114 was right: seq2048 = seq4096 with sliding window eval.
But no clear advantage either way — basically a tie.
Standard eval is worse with seq2048 (fewer steps hurt).
