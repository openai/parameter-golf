# Current State — March 24, 2026 (exp208) — SUBMITTABLE

## First Valid Submission
**1.1127 val_bpb** (TTT@stride=76, 551s eval) | **15.76MB artifact** ✓

## Config
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
- EMA(0.997), Muon WD=0.09
- **QEP GPTQ** int6 quantization, brotli-11 compression
- BigramHash 8192x64
- TTT: online score-first per-window SGD, stride=76

## Key Results
| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | ~1.128 |
| Post-quant roundtrip | 1.1415 (quant gap 0.013) |
| TTT@stride=76 | **1.1127** (submittable, 551s) |
| Artifact size | **15,761,261 bytes (15.76MB)** ✓ |

## Why WD=0.09 not 0.05
WD=0.05 gives better BPP (1.1078) but artifact is 18.3MB — over the 16MB limit.
Only WD≥0.09 produces artifacts that fit under 16MB.
QEP GPTQ gained 0.0003 BPP vs non-QEP (1.1130 → 1.1127).

## Wandb
- Run ID: `i9xcpzv5`
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/i9xcpzv5
- Run name: `size_wd09_qep`

## Script
`clean_train_206_freeze_sweep.py` with WD=0.09, QEP_ENABLED=1
(same as `checkpoints/clean_train_203.py` with different WD + wandb artifact logging)
