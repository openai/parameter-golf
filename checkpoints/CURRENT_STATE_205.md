# Current State — March 24, 2026 (exp205) — RUNNING

## Expected Result
**~1.106 val_bpb** (TTT@stride=64, ~440s eval) | **~15.8MB artifact**

## Config (combined all wins)
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
EVAL_STRIDE=64
MLP_ACTIVATION=leaky2
TTT_ENABLED=1
TTT_MODE=perwindow
TTT_LR=0.002
TTT_EPOCHS=1
TTT_MOMENTUM=0.9
TTT_FREEZE_LAYERS=8
TTT_BATCH_SEQS=128
PRUNE_FRAC=0.0
ROPE_BASE=50000
SWA_ENABLED=0
GPTQ_ENABLED=1
GPTQ_SAMPLES=256
QEP_ENABLED=1
SEED=1337
```

## What's New vs exp203
- **TTT_FREEZE_LAYERS=8** (was 2) — freeze first 8 of 14 layers during TTT, only last 6 adapt
  - Saves ~214s eval time (0.001 BPP cost from freeze sweep)
- **EVAL_STRIDE=64** (was 76) — finer overlap enabled by faster TTT
  - Gains ~0.0004 BPP from better overlap

## Architecture
- Same as exp203: 14L, 512d, 8H/4KV GQA, 3x MLP, leaky_relu(0.5)^2, FA3
- EMA(0.997), Muon WD=0.05, QEP GPTQ int6
- BigramHash 8192x64, brotli-11
- TTT: online score-first per-window SGD, stride=64, freeze 8/14 layers

## Script
`checkpoints/clean_train_205.py` (same code as 203, different env vars)

## Status
RUNNING — launched March 24 ~15:00 UTC. Expected ~25 min total.
