# GatedDeltaNet FLA + Brotli (No TTT)

**val_bpb: 1.01902** (3-seed mean, std 0.0017) | **~15.6 MB** | 8xH100 SXM

## Results

| Seed | Steps | EMA BPB | **Quantized BPB** | Artifact |
|------|-------|---------|-------------------|----------|
| 1337 | 2,400 | 0.998608 | **1.01720** | 15,595,190 |
| 42 | 2,410 | 1.001194 | **1.02054** | 15,602,610 |
| 2025 | 2,408 | 1.001260 | **1.01933** | 15,608,600 |
| **Mean** | **2,406** | **1.000354** | **1.01902 (std 0.0017)** | |

## What Changed

- **GatedDeltaNet (FLA)** K_KVShare_Wider architecture from PR #1687 — O(n) linear attention replacing softmax
- **Brotli-11 compression** instead of zstd-22 — saves ~900KB, all artifacts well under 16MB
- **No TTT** — pure fixed predictor (Track A)

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

## Run Command

```bash
pip install flash-linear-attention==0.4.2 fla-core==0.4.2

ARCH_MODE=K TTT_ENABLED=0 \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **GatedDeltaNet architecture**: PR #1687 by @resouer
- **Flash Linear Attention**: @sustcsonglin (fla-core 0.4.2)
