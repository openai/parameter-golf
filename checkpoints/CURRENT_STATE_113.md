# Current State — March 21, 2026

## Best Result
**1.1187 val_bpb sliding** | **15.95MB artifact** ✅ | exp113

## Script
`checkpoints/clean_train_113.py`

## Config (env vars)
```
NUM_LAYERS=12
BIGRAM_VOCAB_SIZE=16384
BIGRAM_DIM=64
MUON_WD=0.06
ADAM_WD=0.04
MATRIX_LR=0.025
SCALAR_LR=0.025
TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
ITERATIONS=9000
EVAL_STRIDE=64
MLP_ACTIVATION=leaky2
TTT_ENABLED=1
TTT_LR=0.002
TTT_EPOCHS=3
TTT_MOMENTUM=0.9
TTT_FREEZE_LAYERS=2
PRUNE_FRAC=0.02
ROPE_BASE=50000
SWA_ENABLED=0
SEED=1337
```

## What we're using
- **Architecture**: 12-layer transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- **Activation**: leaky_relu(0.5)^2
- **Optimizer**: Muon (WD=0.06, momentum=0.99, warmup 0.92->0.99 over 1500 steps)
- **RoPE**: NTK base=50000
- **SWA**: DISABLED
- **BigramHash**: 16384x64
- **Attention**: FlashAttention 3 (90ms/step vs 107ms without)
- **Training**: seq_len=2048, batch=786K tokens, 600s wallclock, ~90ms/step, 6655 steps
- **Features**: SmearGate, OrthoInit + muP
- **Serialization**: manual (dtype-grouped bytes + JSON header) + zstd-22 (saved 330KB vs torch.save)
- **Quantization**: int6 mixed (MLP+attn=int6, embed=int8)
- **Pruning**: 2% magnitude pruning pre-quantization
- **TTT**: 3 epochs SGD (lr=0.002, momentum=0.9), freeze first 2 blocks
- **Eval**: sliding window stride=64

## Compression comparison (exp113)
```
manual+zstd22:      15,879,232 bytes (total=15,950,410) ← WINNER
manual+zstd22+dict: ~15,900,000 bytes (dict overhead > savings)
torch+zstd22:       ~16,212,000 bytes (what exp110 used)
```
Manual serialization saved ~330KB vs torch.save by grouping tensors by dtype.

## Progression
| Exp | Key change | Sliding val_bpb | Delta | Artifact | Fits? |
|-----|-----------|----------------|-------|----------|-------|
| 094 | PR198 baseline | 1.1374 | — | 15.69MB | Yes |
| 095 | + leaky2 | 1.1355 | -0.002 | 15.71MB | Yes |
| 097 | + TTT + prune | 1.1314 | -0.004 | 15.72MB | Yes |
| 103 | + 12L + WD=0.06 | 1.1288 | -0.003 | 15.87MB | Yes |
| 105 | + ROPE_BASE=50000 | 1.1280 | -0.001 | 15.88MB | Yes |
| 106 | - SWA (removed!) | 1.1267 | -0.001 | 15.91MB | Yes |
| 108 | + BigBigram 16384x64 | 1.1231 | -0.004 | 15.91MB | Yes |
| 110 | + FlashAttention 3 | 1.1187 | -0.004 | 16.28MB | No |
| **113** | **+ manual serialization** | **1.1187** | **0** | **15.95MB** | **Yes** |

## Competition context
- PR254 record: 1.1303 val_bpb
- **We beat the record by 0.0116: 1.1187 vs 1.1303** ✅
- **Artifact fits under 16MB** ✅ (50KB headroom)
