# Current State -- March 23, 2026

## Best Result
**1.1158 val_bpb sliding** | **15.80MB artifact** | exp187

## Script
`checkpoints/clean_train_187.py`

## Config (env vars)
```
EMA_ENABLED=1
EMA_DECAY=0.997
NUM_LAYERS=13
BIGRAM_VOCAB_SIZE=8192
BIGRAM_DIM=64
MUON_WD=0.07
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

## What changed from exp182b (previous best at 1.1173)
1. **NUM_LAYERS=13** (was 12): Extra layer adds capacity. 7% slower per step (97.5ms vs 90.9ms) but each step learns more. Only 6152 steps vs 6610.
2. **MUON_WD=0.07** (was 0.05): Higher WD compensates for more params — keeps artifact compressible. The extra layer's capacity outweighs the WD quality penalty.

## What we're using
- **Architecture**: **13-layer** transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- **Activation**: leaky_relu(0.5)^2
- **Optimizer**: Muon (WD=0.07, momentum=0.99, warmup 0.92->0.99 over 1500 steps) + Adam (WD=0.02)
- **EMA**: decay=0.997, from step 0, every step, float32 accumulation
- **RoPE**: NTK base=50000
- **SWA**: DISABLED
- **BigramHash**: 8192x64
- **Attention**: FlashAttention 3 (~97.5ms/step with 13L)
- **Training**: seq_len=2048, batch=786K tokens, 600s wallclock, ~97.5ms/step, 6152 steps
- **Features**: SmearGate, OrthoInit + muP
- **Serialization**: manual (dtype-grouped bytes + JSON header) + **brotli-11**
- **Quantization**: int6 mixed (MLP+attn=int6, embed=int8)
- **Pruning**: DISABLED
- **TTT**: 3 epochs SGD (lr=0.002, momentum=0.9), freeze first 2 blocks
- **Eval**: sliding window stride=64

## Compression comparison (exp187)
```
manual+brotli11:    15,720,702 bytes (total=15,798,567)  <-- WINNER (201KB headroom)
torch+brotli11:     15,730,131 bytes (total=15,807,996)
```

## Progression
| Exp | Key change | Sliding val_bpb | Delta | Artifact | Fits? |
|-----|-----------|----------------|-------|----------|-------|
| 113 | Previous checkpoint | 1.1187 | -- | 15.95MB | Yes |
| 182b | + EMA + WD=0.05 + brotli | 1.1173 | -0.0014 | 15.92MB | Yes |
| **187** | **+ 13L + WD=0.07** | **1.1158** | **-0.0015** | **15.80MB** | **Yes** |

## Competition context
- Official leaderboard SOTA: 1.1428 (thwu1)
- **We beat the record by 0.0270: 1.1158 vs 1.1428**
- Best known PR: PR#414 at 1.1233, PR#254 at 1.1303
- **We beat ALL known submissions by significant margin**
- Dependencies: requires `brotli` package (pip install brotli)
