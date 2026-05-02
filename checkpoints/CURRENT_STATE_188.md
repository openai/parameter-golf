# Current State -- March 23, 2026

## Best Result
**1.1155 val_bpb sliding** | **15.83MB artifact** | exp188

## Script
`checkpoints/clean_train_188.py`

## Config (env vars)
```
EMA_ENABLED=1
EMA_DECAY=0.997
NUM_LAYERS=14
BIGRAM_VOCAB_SIZE=8192
BIGRAM_DIM=64
MUON_WD=0.09
ADAM_WD=0.02
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3500
ITERATIONS=9000
EVAL_STRIDE=64
MLP_ACTIVATION=leaky2
TTT_ENABLED=1
TTT_LR=0.002
TTT_EPOCHS=3
TTT_MOMENTUM=0.9
TTT_FREEZE_LAYERS=2
PRUNE_FRAC=0.0
ROPE_BASE=50000
SWA_ENABLED=0
SEED=1337
```

## Architecture
- **14-layer** transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- leaky_relu(0.5)^2, FlashAttention 3
- EMA(0.997), Muon WD=0.09, Adam WD=0.02
- ~105ms/step, 5707 steps in 600s
- BigramHash 8192x64, brotli-11 compression
- TTT: 3 epochs SGD (lr=0.002), online sliding stride=64

## Depth scaling results (all with EMA + brotli)
| Layers | WD | BPP | Artifact | Steps |
|--------|-----|-----|----------|-------|
| 12 | 0.05 | 1.1173 | 15.92MB | 6610 |
| 13 | 0.07 | 1.1158 | 15.80MB | 6152 |
| **14** | **0.09** | **1.1155** | **15.83MB** | **5707** |

Each extra layer costs ~7% speed (fewer steps) but adds ~0.003 BPP per-step improvement.
