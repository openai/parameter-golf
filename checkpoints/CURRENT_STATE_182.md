# Current State -- March 23, 2026

## Best Result
**1.1173 val_bpb sliding** | **15.92MB artifact** | exp182b

## Script
`checkpoints/clean_train_182.py`

## Config (env vars)
```
EMA_ENABLED=1
EMA_DECAY=0.997
NUM_LAYERS=12
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
TTT_LR=0.002
TTT_EPOCHS=3
TTT_MOMENTUM=0.9
TTT_FREEZE_LAYERS=2
PRUNE_FRAC=0.0
ROPE_BASE=50000
SWA_ENABLED=0
SEED=1337
```

## What changed from exp113 (previous best at 1.1187)
1. **EMA (decay=0.997)**: Exponential moving average of weights from step 0. Smooths weights for better quantization. -0.002 BPP.
2. **MUON_WD=0.05** (was 0.06): Lower weight decay gives better BPP. Requires better compression to fit.
3. **ADAM_WD=0.02** (was 0.04): Lower Adam WD for embedding/scalar params. -0.0002 BPP.
4. **WARMDOWN_ITERS=3500** (was 3000): Longer warmdown. -0.0002 BPP.
5. **BIGRAM_VOCAB_SIZE=8192** (was 16384): Halved bigram vocab to save artifact space for EMA + lower WD.
6. **PRUNE_FRAC=0.0** (was 0.02): Pruning hurts EMA (makes weights less smooth).
7. **Brotli compression** (was zstd-22): Brotli-11 saves 364KB vs zstd-22 on our int6 weights. This is the key that makes WD=0.05 fit under 16MB.

## What we're using
- **Architecture**: 12-layer transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- **Activation**: leaky_relu(0.5)^2
- **Optimizer**: Muon (WD=0.05, momentum=0.99, warmup 0.92->0.99 over 1500 steps) + Adam (WD=0.02)
- **EMA**: decay=0.997, from step 0, every step, float32 accumulation
- **RoPE**: NTK base=50000
- **SWA**: DISABLED (incompatible with EMA)
- **BigramHash**: 8192x64 (halved from 16384 to fit artifact)
- **Attention**: FlashAttention 3 (~90ms/step)
- **Training**: seq_len=2048, batch=786K tokens, 600s wallclock, ~90.8ms/step, 6610 steps
- **Features**: SmearGate, OrthoInit + muP
- **Serialization**: manual (dtype-grouped bytes + JSON header) + **brotli-11** (saves 364KB vs zstd-22)
- **Quantization**: int6 mixed (MLP+attn=int6, embed=int8)
- **Pruning**: DISABLED (hurts EMA)
- **TTT**: 3 epochs SGD (lr=0.002, momentum=0.9), freeze first 2 blocks
- **Eval**: sliding window stride=64

## Compression comparison (exp182b)
```
manual+brotli11:    15,846,542 bytes (total=15,924,407)  <-- WINNER
torch+brotli11:     15,856,436 bytes (total=15,934,301)
torch+zstd22+dict:  ~16,137,000 bytes (over limit)
torch+zstd22:       ~16,211,000 bytes (over limit)
manual+zstd22+dict: ~16,308,000 bytes (over limit)
```
Brotli-11 compression is the breakthrough that makes WD=0.05 fit under 16MB.

## Progression from exp113
| Exp | Key change | Sliding val_bpb | Delta | Artifact | Fits? |
|-----|-----------|----------------|-------|----------|-------|
| 113 | Previous best (baseline) | 1.1187 | -- | 15.95MB | Yes |
| 173 | + EMA + bigram8k | 1.1199 | +0.001 | 15.57MB | Yes |
| 176 | + ADAM_WD=0.02 + WD3500 | 1.1189 | -0.001 | 15.43MB | Yes |
| 179 | + MUON_WD=0.055 | 1.1182 | -0.001 | 15.74MB | Yes |
| 180 | + MUON_WD=0.05 (no brotli) | 1.1176 | -0.001 | 16.09MB | No |
| **182b** | **+ brotli compression** | **1.1173** | **-0.000** | **15.92MB** | **Yes** |

## Competition context
- Previous best (exp113): 1.1187 val_bpb
- **Improvement: -0.0014 BPP (1.1187 -> 1.1173)**
- Artifact fits under 16MB with 75KB headroom
- Dependencies: requires `brotli` package (pip install brotli)

## Dead ends tested this session
- Partial RoPE 32/64: +0.002 worse
- Simple XSA (mean-subtract): invalid (leaks future info)
- PRP (element-wise modulated random matrix): +0.027 worse at dim=512
- Butterfly FFT projection: 2.6x slower, killed
- BIGRAM_DIM=48: -0.002 worse than dim=64
