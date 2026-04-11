# 11L XSA-all + 7-gram Eval Cache

## Results

| Seed | val_bpb |
|------|---------|
| 42   | 1.0467  |
| 1337 | 1.0470  |
| 2024 | 1.0457  |
| **Mean** | **1.0465** |
| Std  | 0.0007  |

Artifact: 13.99 MB. Train: 600s on 8xH100 SXM. Eval: ~116s.

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

## 7-gram Eval Cache

- Order 7, fixed alpha=0.40, 4M hash buckets, min_count=2
- Score-first backward-looking, deterministic
- No TTT, no gradient updates

## Acknowledgments

Architecture based on community techniques from PRs #609, #549, and others.
