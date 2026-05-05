# Current State — March 24, 2026 (exp203)

## Best Submission-Ready Result
**1.1075 val_bpb** (TTT@stride=76, 551s eval) | **15.83MB artifact**

## Config
```
EMA_ENABLED=1
EMA_DECAY=0.997
NUM_LAYERS=14
BIGRAM_VOCAB_SIZE=8192
BIGRAM_DIM=64
MUON_WD=0.05
ADAM_WD=0.02
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3500
ITERATIONS=9000
EVAL_STRIDE=76
MLP_ACTIVATION=leaky2
TTT_ENABLED=1
TTT_MODE=perwindow
TTT_LR=0.002
TTT_EPOCHS=1
TTT_MOMENTUM=0.9
TTT_FREEZE_LAYERS=2
TTT_BATCH_SEQS=128
PRUNE_FRAC=0.0
ROPE_BASE=50000
SWA_ENABLED=0
GPTQ_ENABLED=1
GPTQ_SAMPLES=256
QEP_ENABLED=1
SEED=1337
```

## Architecture
- **14-layer** transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- leaky_relu(0.5)^2, FlashAttention 3
- EMA(0.997), Muon WD=0.05
- **QEP GPTQ** int6 quantization (sequential block calibration with error propagation)
- BigramHash 8192x64, brotli-11 compression
- TTT: online score-first per-window SGD, stride=76

## Key Results
| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | ~1.120 |
| Post-quant roundtrip | 1.1327 (quant gap 0.012) |
| TTT@stride=76 | **1.1075** (submission-ready, 551s) |
| Rescore@stride=64 | 1.1060 (over time budget at 638s) |

## What Changed vs exp199
- WD: 0.09 → **0.05** (optimal per U-shaped sweep)
- QEP GPTQ: reduces quant gap 0.015 → 0.012
- Combined: -0.003 BPP improvement
