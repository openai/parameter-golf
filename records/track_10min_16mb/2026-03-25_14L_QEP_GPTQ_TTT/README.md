# 14L QEP GPTQ + Per-Window SGD TTT

**val_bpb = 1.1127** (seed 1337) | **15.76 MB** | 8xH100 SXM | **551s eval**

## Architecture

- 14L, 512d, 8H/4KV (GQA), MLP 3x (h=1536), leaky_relu(0.5)²
- U-Net skip connections (7 encoder + 7 decoder)
- BigramHash(8192, dim=64), FlashAttention 3
- EMA(0.997), Muon WD=0.09, warmdown 3500 steps
- ~26.8M params, 9000 training iterations (5700 in 600s wallclock)

## Quantization

- **QEP-aware GPTQ** (Quantization Error Propagation, arxiv:2504.09629)
- Sequential block calibration: quantize blocks 0→13, each block's Hessian uses partially-quantized model outputs
- Int6 per-row quantization for attention + MLP weights, int8 for embeddings
- Percentile clipping search (99.9%, 99.95%, 99.99%, 99.999%, 100%)
- Brotli-11 compression → 15.76MB artifact

## Test-Time Training

- Online per-window SGD TTT (legal per issue #402 ruling)
- Score tokens first, then adapt on scored tokens (causal, no future leakage)
- SGD lr=0.002, momentum=0.9, stride=76, 1 epoch per window
- Freeze first 2 of 14 layers during TTT
- 3188 adapt steps during eval, 551s total

## Key Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | ~1.128 |
| Post-quant roundtrip | 1.1415 (quant gap 0.013) |
| **TTT@stride=76** | **1.1127** |
| Artifact | 15,761,261 bytes |
| Eval time | 551s |

## Reproduction

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=1337 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
NUM_LAYERS=14 BIGRAM_VOCAB_SIZE=8192 BIGRAM_DIM=64 \
MUON_WD=0.09 ADAM_WD=0.02 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 \
EVAL_STRIDE=76 MLP_ACTIVATION=leaky2 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 \
ROPE_BASE=50000 SWA_ENABLED=0 \
GPTQ_ENABLED=1 GPTQ_SAMPLES=256 QEP_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Platform

8×H100 SXM, PyTorch 2.8.0+cu128

## Credits

- Base: modded-nanogpt (KellerJordan)
- QEP: arxiv:2504.09629
- TTT approach: issue #402 ruling
