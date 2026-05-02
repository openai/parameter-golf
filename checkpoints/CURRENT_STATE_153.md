# Exp153: MUON_WD=0.04 — March 22, 2026

## Result
**1.1167 val_bpb sliding (online TTT)** | **17.49MB artifact** | OVER 16MB LIMIT

## Change from baseline (exp146 = 1.1201)
- MUON_WD: 0.06 -> **0.04** (matches top PRs)
- Everything else identical to exp113 config + online TTT

## Key Metrics
| Metric | Baseline (exp146) | Exp153 |
|--------|-------------------|--------|
| Sliding BPB (online TTT) | 1.1201 | **1.1167** |
| Roundtrip BPB | 1.1497 | 1.1431 |
| Artifact | 15.98MB | **17.49MB** |
| Steps | 6660 | ~6650 |
| Step time | 90ms | 90ms |

## Verdict
**BEST QUALITY SO FAR (-0.003 BPB)** but artifact 1.5MB over limit. Lower WD = better BPB but worse compression. Need to combine with INT5_MLP or higher pruning to fit 16MB. 5% pruning tested (exp154) — did NOT help compression (17.50MB).

## To make it fit
- INT5_MLP=1 would save ~1.2MB (need to verify)
- Or MUON_WD=0.05 as compromise (untested)
- Or combine with ADAM_WD=0.02 + MUON_WD=0.05

## Config
```
NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=16384 BIGRAM_DIM=64
MUON_WD=0.04 ADAM_WD=0.04
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000 ITERATIONS=9000 EVAL_STRIDE=64
MLP_ACTIVATION=leaky2 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3
TTT_MOMENTUM=0.9 TTT_FREEZE_LAYERS=2 PRUNE_FRAC=0.02
ROPE_BASE=50000 SWA_ENABLED=0 SEED=1337
```

## Script
`checkpoints/clean_train_153_muonwd04.py` (= clean_train_113_online_ttt.py)
