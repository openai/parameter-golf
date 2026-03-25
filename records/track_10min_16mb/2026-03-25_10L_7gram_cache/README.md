# 10L + 7-gram Eval Cache

## Results

| Seed | val_bpb | Eval time |
|------|---------|-----------|
| 42   | 1.0531  | 195s      |
| 1337 | 1.0814  | ~195s     |
| 2024 | 1.0805  | ~195s     |
| **Mean** | **1.0717** | |
| Std  | 0.0160  | |

Artifact size: 15.75 MB. Train: 600s on 8xH100 SXM. Eval: ~195s.

## Architecture

- 10 layers, 512 dim, 8/4 GQA heads
- 3x MLP with LeakyReLU(0.5)^2
- BigramHash (10240 buckets, 128 dim)
- SmearGate, value residual, gated attention
- U-Net skip connections, tied embeddings
- Mixed int5 (MLP) / int6 (attention) quantization + zstd-22
- 5% magnitude pruning, EMA (0.995)
- Muon (lr=0.02, mom=0.99, WD=0.04) + AdamW

## 7-gram Eval Cache

Backward-looking n-gram cache applied during sliding-window evaluation:

- Order 7, fixed alpha=0.40, 4M hash buckets, min_count=2
- XOR-based context hashing with prime multipliers
- Score-first: each token scored by model before entering cache
- Deterministic: identical results given same weights and data
- No TTT, no gradient updates at eval time
- Adds zero learned parameters
