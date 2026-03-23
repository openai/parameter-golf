# 10L Optimized + LoRA TTT

10-layer transformer with per-document LoRA adaptation at eval time.

## Results

| Seed | Base val_bpb | TTT val_bpb |
|------|-------------|-------------|
| 42   | 1.1485      | 1.1039      |

Artifact size: 15.75 MB

Note: TTT eval currently exceeds 10-min eval cap (~29 min). Working on speedup. Base model score (1.1485) is within time limits.

## Architecture

- 10 layers, 512 dim, 8/4 GQA heads
- 3x MLP, improved activations
- Hash-based n-gram embeddings
- U-Net skip connections, tied embeddings
- Mixed int5/int6 quantization + zstd-22

## TTT

- Rank-8 LoRA on Q, V projections + LM head
- Per-document reset
- Adam optimizer, 3 epochs per doc
- Score-then-train per chunk

## Training

- Muon + AdamW, EMA averaging
- 10 minutes on 8xH100 SXM
