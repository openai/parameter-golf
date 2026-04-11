# 11L BackoffNgramMixer

## Results

| Seed | val_bpb | Eval time |
|------|---------|-----------|
| 42   | 0.6672  | 512s      |
| 1337 | 0.6673  | ~512s     |
| 2024 | 0.6667  | ~512s     |
| **Mean** | **0.6671** | |
| Std  | 0.0003  | |

Artifact: ~16.0 MB. Train: 600s on 8xH100 SXM. Eval: ~512s.

## Architecture

- 11 layers, 512 dim, 8/8 full MHA heads
- XSA-all, LeakyReLU(0.5)^2, 3.5x MLP
- BigramHash, SmearGate, Value Residual, Gated Attention
- int5 quantization + zstd compression
- EMA, Tight SWA, Soft-Round QAT

## BackoffNgramMixer

GPU-vectorized multi-order n-gram backoff (orders 2-7) with entropy-adaptive alpha mixing. Score-first backward-looking cache with per-token entropy gating.

## Acknowledgments

Architecture and mixer based on community techniques.
