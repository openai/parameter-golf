# 11L XSA-all + Multi-order Backoff 7-gram

## Results

| Seed | val_bpb |
|------|---------|
| 42   | 0.9920  |
| 1337 | 0.9912  |
| 2024 | 0.9920  |
| **Mean** | **0.9917** |
| Std  | 0.0005  |

Artifact: 13.99 MB. Train: 600s on 8xH100 SXM. Eval: ~150s.

## Architecture

- 11 layers, 512 dim, 8/4 GQA heads
- XSA (Exclusive Self-Attention) on all 11 layers
- LeakyReLU(0.5)^2 MLP, 3x expansion
- BigramHash (10240, 128d), SmearGate
- Value Residual, Gated Attention
- U-Net skip, tied embeddings, LN scaling 1/sqrt(layer+1)
- GPTQ-lite int6 quantization + zstd-22
- EMA(0.997), Tight SWA, Late QAT
- Muon (lr=0.025, mom=0.99, WD=0.04) + AdamW

## Multi-order Backoff N-gram Eval Cache

- Orders 2 through 7, with highest-order-first backoff
- Fixed alpha=0.40, 4M hash buckets per order, min_count=2
- Score-first: each token scored before entering any cache table
- Deterministic, no TTT, no gradient updates
- Cache tables maintained independently per order to avoid cross-order hash collision

## Acknowledgments

Architecture based on community techniques.
