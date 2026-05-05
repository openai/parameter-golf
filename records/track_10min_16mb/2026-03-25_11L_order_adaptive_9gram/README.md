# 11L XSA-all + Order-Adaptive 9-gram Backoff

## Results

| Seed | val_bpb |
|------|---------|
| 42   | 0.9067  |
| 1337 | 0.9059  |
| 2024 | 0.9050  |
| **Mean** | **0.9059** |
| Std  | 0.0009  |

Artifact: 13.99 MB. Train: 600s on 8xH100 SXM. Eval: ~150s.

## Architecture

- 11 layers, 512 dim, 8/4 GQA heads
- XSA (Exclusive Self-Attention) on all 11 layers
- LeakyReLU(0.5)^2, 3x MLP
- BigramHash (10240, 128d), SmearGate
- Value Residual, Gated Attention, LN scaling
- GPTQ-lite int6 + zstd-22
- EMA(0.997), Tight SWA, Late QAT

## Order-Adaptive N-gram Eval Cache

Multi-order backoff (2 through 9-gram) with order-adaptive entropy gating:

- Higher-order matches (9-gram) get lower entropy threshold, trusting n-gram even when model is moderately confident
- Lower-order matches (2-gram) require high model uncertainty before trusting n-gram
- Formula: `center = 3.0 - 0.25 * (matched_order - 2)`, `alpha = 0.05 + 0.55 * sigmoid(2 * (H - center))`
- Separate hash tables per order (no cross-order collision)
- Score-first, deterministic, no TTT
