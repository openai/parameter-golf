# 10L + Batched LoRA TTT

10-layer transformer with batched per-document LoRA adaptation at eval time.

## Results

| Seed | Base val_bpb | TTT val_bpb | TTT time |
|------|-------------|-------------|----------|
| 42   | 1.1476      | 1.1160      | 495s     |

Artifact size: 15.75 MB. Single seed, more seeds coming.

## Architecture

- 10 layers, 512 dim, 8/4 GQA heads
- 3x MLP with LeakyReLU(0.5)^2
- BigramHash (10240 buckets, 128 dim)
- SmearGate, value residual, gated attention
- U-Net skip connections, tied embeddings
- Mixed int5 (MLP) / int6 (attention) quantization + zstd-22
- 5% magnitude pruning
- EMA weight averaging (decay=0.995)

## Training

- Muon optimizer (lr=0.02, momentum=0.99, WD=0.04) + AdamW
- 10 minutes on 8xH100 SXM, ~5800 steps

## LoRA TTT

- Rank-8 LoRA on Q, V projections + LM head (all layers)
- 64 documents batched in parallel
- Per-document fresh initialization + optimizer reset
- Adam (lr=0.01, betas 0.9/0.95), 256-token chunks, 3 epochs
- Score on final epoch only
- Documents split by BOS boundaries
