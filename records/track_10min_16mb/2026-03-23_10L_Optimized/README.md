# 10L + Batched LoRA TTT

10-layer transformer with batched per-document LoRA adaptation.

## Results

| Seed | Base val_bpb | TTT val_bpb | TTT time |
|------|-------------|-------------|----------|
| 42   | 1.1476      | 1.1160      | 495s     |

Artifact size: 15.75 MB
Train time: 600s (10 min)
Eval time (TTT): 495s + 180s sliding eval = 675s (11.3 min, needs minor optimization)

## Architecture

- 10 layers, 512 dim, 8/4 GQA heads
- 3x MLP, improved activations
- Hash-based n-gram embeddings (10240 buckets)
- U-Net skip connections, tied embeddings
- Mixed int5/int6 quantization + zstd-22, 5% magnitude pruning
- EMA weight averaging (decay=0.995)

## LoRA TTT

- Rank-8 LoRA on Q, V projections + LM head (all layers)
- Batched processing: 64 documents in parallel
- Per-document reset, Adam (lr=0.01, betas 0.9/0.95)
- 256-token chunks, 3 epochs per document
- Score on final epoch only
- Documents found via BOS token boundaries
