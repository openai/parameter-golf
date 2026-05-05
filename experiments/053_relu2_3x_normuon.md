# Experiment 053: relu² 3x h=1536 + NorMuon + seq4096 (A/B vs SwiGLU)

## Status: COMPLETED

## Config
- USE_SWIGLU=0, MLP_MULT=3 (relu² 3x, h=1536)
- NorMuon, int6 QAT, seq4096, batch393K
- No SWA, no LAWA, no grad clip

## Results
| Metric | Value |
|--------|-------|
| Steps | 10,756 @ 55.8ms/step |
| Artifact | 16,173,646 bytes ❌ (174KB over) |
| **Standard eval** | **1.1754 BPB** |
| **Sliding eval s64** | **1.1639 BPB** ← NEW BEST |

## A/B Comparison: relu² vs SwiGLU (same everything else)
| | 049 SwiGLU h=1024 | 053 relu² h=1536 | Diff |
|--|-------------------|------------------|------|
| Steps | 10,410 | 10,756 | +346 (3% more) |
| ms/step | 57.6 | 55.8 | -1.8ms (3% faster) |
| Standard | 1.1805 | **1.1754** | **-0.005** |
| Sliding | 1.1685 | **1.1639** | **-0.0046** |

## KEY FINDING
**relu² 3x beats SwiGLU at this scale with int6 QAT.**
SwiGLU was our early winner at small scale (exp027-030) but at 21.8M params with int6:
- relu² is 3% faster (2 matrices vs 3)
- relu² quantizes better (fewer, larger matrices)
- relu² gets more steps in 10 min
- relu² has better per-step quality at this scale

**DECISION: Switch all future experiments to relu² 3x. Drop SwiGLU.**

## wandb
- Run name: 053_relu2_3x_normuon
