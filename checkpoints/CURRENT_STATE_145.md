# Current State — March 21, 2026 (exp145)

## Best Result
**1.1185 val_bpb sliding** | **15.96MB artifact** | exp145 (GPTQ-lite)

## Improvement over previous best
- Previous: 1.1187 (exp113)
- Current: 1.1185 (exp145)
- Delta: **-0.0002 BPB** (from smarter quantization alone)

## Script
`experiments/clean_train_145_gptqlite.py`

## Config (env vars) — identical to exp113 except GPTQ-lite quantization
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

## What changed from exp113
- **GPTQ-lite quantization**: Instead of simple per-row max scaling, searches 6 clip ratios (0.85, 0.90, 0.93, 0.96, 0.99, 1.0) per weight matrix and picks the one minimizing reconstruction error. Zero training cost — only affects post-training quantization.
- **Compression**: GPTQ-lite changes weight distributions such that `manual+zstd22+dict` (15.96MB) is now best. Without dict, `manual+zstd22` is 16.49MB (over limit!).

## What we're using
- **Architecture**: 12-layer transformer, 512 dim, 8 heads, 4 KV heads (GQA), 3x MLP (h=1536)
- **Activation**: leaky_relu(0.5)^2
- **Optimizer**: Muon (WD=0.06, momentum=0.99, warmup 0.92->0.99 over 1500 steps)
- **RoPE**: NTK base=50000
- **SWA**: DISABLED
- **BigramHash**: 16384x64
- **Attention**: FlashAttention 3 (90ms/step)
- **Training**: seq_len=2048, batch=786K tokens, 600s wallclock, ~90ms/step, 6659 steps
- **Features**: SmearGate, OrthoInit + muP
- **Serialization**: manual (dtype-grouped bytes + JSON header) + zstd-22 with dictionary
- **Quantization**: GPTQ-lite int6 (optimal clip search) for MLP+attn, int8 for embed
- **Pruning**: 2% magnitude pruning pre-quantization
- **TTT**: 3 epochs SGD (lr=0.002, momentum=0.9), freeze first 2 blocks
- **Eval**: sliding window stride=64

## Key metrics (seed=1337)
| Metric | exp113 (baseline) | exp145 (GPTQ-lite) |
|--------|-------------------|-------------------|
| Steps | 6660 | 6659 |
| Pre-quant val_bpb | 1.1329 | 1.1326 |
| Post-quant roundtrip | 1.1414 | 1.1412 |
| **Sliding BPB** | **1.1187** | **1.1185** |
| Artifact size | 15.95MB | 15.96MB |
| TTT epochs | 1.9339/1.9301/1.9286 | 1.9333/1.9296/1.9282 |

## Compression comparison (exp145)
```
manual+zstd22+dict: 15,883,153 bytes (total=15,955,451) <- WINNER
manual+zstd22:      16,417,278 bytes (total=16,489,576) <- OVER LIMIT!
manual+lzma9:       16,576,868 bytes (total=16,649,166)
torch+zstd22:       16,655,423 bytes (total=16,727,721)
```
NOTE: GPTQ-lite makes manual+zstd22 go OVER 16MB. Dictionary compression is now required.

## Still need
- Multi-seed validation (seed=42, seed=2026) to confirm statistical significance
