# GatedDeltaNet FLA + Score-First TTT + Brotli

**val_bpb: 1.00980** (3-seed mean, std 0.0015) | **~15.6 MB** | 8xH100 SXM

## Results

| Seed | Steps | EMA BPB | Pre-TTT BPB | **Post-TTT BPB** | TTT Gain | Artifact |
|------|-------|---------|-------------|-----------------|----------|----------|
| 1337 | 2,400 | 0.998608 | 1.01720 | **1.00803** | -0.00917 | 15,595,190 |
| 42 | 2,410 | 1.001194 | 1.02054 | **1.01069** | -0.00986 | 15,602,610 |
| 2025 | 2,408 | 1.001260 | 1.01933 | **1.01067** | -0.00866 | 15,608,600 |
| **Mean** | **2,406** | **1.000354** | **1.01902** | **1.00980 (std 0.0015)** | **-0.00923** | |

## What Changed

- **GatedDeltaNet (FLA)** K_KVShare_Wider architecture from PR #1687 — O(n) linear attention replacing softmax
- **Brotli-11 compression** instead of zstd-22 — saves ~900KB, all artifacts well under 16MB
- **Score-first TTT** (SGD lr=0.005, momentum=0.9, 3 epochs, freeze first 2 blocks) adapted from PR #461

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 10 GDN (544d, 8H) |
| KV Sharing | Stride 2 |
| MLP | 3x width |
| BigramHash | 3072 x 112 + trigram |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Late QAT | Threshold 0.15 |
| Quantization | Int6 matrices + Int8 embeddings |
| Compression | Brotli quality 11 |
| Optimizer | Muon (matrices) + Adam (scalars/embeds) |

## TTT Protocol

Legal score-first TTT per Issue #1017:

1. Val tokens split into 32K-token chunks
2. For each chunk: **SCORE** under `torch.inference_mode()`, then **TRAIN** on already-scored tokens
3. SGD(lr=0.005, momentum=0.9), 3 epochs, freeze first 2 blocks, cosine LR decay
4. Last chunk scored but never trained on

## Run Command

```bash
pip install flash-linear-attention==0.4.2 fla-core==0.4.2

ARCH_MODE=K TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=2 TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **GatedDeltaNet architecture**: PR #1687 by @resouer
- **Flash Linear Attention**: @sustcsonglin (fla-core 0.4.2)
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
