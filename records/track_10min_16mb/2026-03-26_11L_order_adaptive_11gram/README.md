# 11L XSA-all + Order-Adaptive 11-gram Backoff

## Results

| Seed | val_bpb |
|------|---------|
| 42   | 0.8883  |
| 1337 | 0.8886  |
| 2024 | 0.8875  |
| **Mean** | **0.8881** |
| Std  | 0.0006  |

Artifact: 13.99 MB. Train: 600s on 8xH100 SXM. Eval: ~160s.

## Architecture

- 11 layers, 512 dim, 8/4 GQA heads
- XSA-all, LeakyReLU(0.5)^2, 3x MLP
- BigramHash (10240, 128d), SmearGate
- Value Residual, Gated Attention, LN scaling
- GPTQ-lite int6 + zstd-22
- EMA(0.997), Tight SWA, Late QAT

## Order-Adaptive N-gram Eval Cache

- Orders 2 through 11, highest-order-first backoff
- Order-adaptive entropy gating: higher-order matches trust n-gram at lower model uncertainty
- Separate hash tables per order
- Score-first, deterministic, no TTT
