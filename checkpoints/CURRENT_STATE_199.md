# Current State -- March 23, 2026

## Best Result
**~1.1107 val_bpb sliding (projected)** | **~15.83MB artifact** | exp199

## What's New (vs exp188)
- **Real GPTQ quantization** (Hessian-aware weight quantization)
- Quantization gap reduced from +0.0202 → +0.0149 (26% reduction)
- Pre-quant val_bpb: 1.1268, Post-quant int6 val_bpb: 1.1417

## Script
`checkpoints/clean_train_199.py`

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
GPTQ_ENABLED=1
GPTQ_SAMPLES=256
SEED=1337
```

## Architecture
- **14-layer** transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- leaky_relu(0.5)^2, FlashAttention 3
- EMA(0.997), Muon WD=0.09, Adam WD=0.02
- GPTQ int6 quantization (256 calibration samples, block_size=128)
- BigramHash 8192x64, brotli-11 compression
- TTT: online score-first, SGD (lr=0.002), sliding stride=64

## Key Results
| Metric | exp188 (naive int6) | exp199 (GPTQ int6) | Delta |
|--------|-------------------|-------------------|-------|
| Pre-quant val_bpb | 1.1258 | 1.1268 | +0.0010 |
| Post-quant int6 val_bpb | 1.1460 | 1.1417 | **-0.0043** |
| Quantization gap | +0.0202 | +0.0149 | **-0.0053** |
| Final SW+TTT val_bpb | 1.1155 | ~1.1107 (proj) | ~-0.0048 |

## Known Issues
- `TTT_EPOCHS=3` is configured but the code only does 1 adapt step per batch (parameter unused in TTT function)
- GPTQ calibration adds ~30s to post-training quantization phase
