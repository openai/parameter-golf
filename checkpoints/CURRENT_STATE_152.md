# Exp152: ADAM_WD=0.02 — March 22, 2026

## Result
**1.1199 val_bpb sliding (online TTT)** | **16.00MB artifact** | Fits 16MB

## Change from baseline (exp146 = 1.1201)
- ADAM_WD: 0.04 -> **0.02** (halved)
- Everything else identical to exp113 config + online TTT

## Key Metrics
| Metric | Baseline (exp146) | Exp152 |
|--------|-------------------|--------|
| Sliding BPB (online TTT) | 1.1201 | **1.1199** |
| Roundtrip BPB | 1.1497 | 1.1493 |
| Artifact | 15.98MB | 16.00MB |
| Steps | 6660 | 6653 |
| Step time | 90ms | 90ms |

## Verdict
Marginal improvement (-0.0002). Within noise but directionally positive. Lower Adam WD gives embeddings/scalars more freedom to learn.

## Config
```
NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=16384 BIGRAM_DIM=64
MUON_WD=0.06 ADAM_WD=0.02
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000 ITERATIONS=9000 EVAL_STRIDE=64
MLP_ACTIVATION=leaky2 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3
TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 PRUNE_FRAC=0.02
ROPE_BASE=50000 SWA_ENABLED=0 SEED=1337
```

## Script
`checkpoints/clean_train_152_adamwd02.py` (= clean_train_113_online_ttt.py)
