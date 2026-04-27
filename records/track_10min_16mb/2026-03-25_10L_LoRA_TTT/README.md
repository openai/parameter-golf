# 10L + Batched LoRA TTT

## Results

| Seed | Base val_bpb | TTT val_bpb | TTT time |
|------|-------------|-------------|----------|
| 42   | 1.1476      | 1.1160      | 495s     |
| 1337 | 1.1540      | 1.1210      | 496s     |
| 2024 | 1.1504      | 1.1170      | 497s     |
| **Mean** | **1.1507** | **1.1180** | **496s** |
| Std  | 0.0032      | 0.0026      |          |

Artifact size: 15.75 MB. Train time: 600s. Eval time (TTT): ~496s.

## Architecture

- 10 layers, 512 dim, 8/4 GQA heads, head_dim=64
- 3x MLP with LeakyReLU(0.5)^2 activation
- BigramHash (10240 buckets, 128 dim)
- SmearGate, value residual connections, per-head gated attention
- U-Net encoder-decoder skip connections
- Tied embeddings (1024 vocab)
- Mixed int5 (MLP) / int6 (attention) quantization + zstd-22
- 5% magnitude pruning
- EMA weight averaging (decay=0.995)
- Muon optimizer (lr=0.02, momentum=0.99, WD=0.04) + AdamW

## LoRA TTT

- Rank-8 LoRA on Q, V projections + LM head across all 10 layers
- 64 documents batched in parallel
- Per-document fresh initialization and optimizer reset
- Adam optimizer (lr=0.01, betas 0.9/0.95)
- 256-token chunks, 3 epochs per document
- Score on final epoch only
- Documents identified by BOS token boundaries
- Short documents (<512 tokens) scored without TTT
