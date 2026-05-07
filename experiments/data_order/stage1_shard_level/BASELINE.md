# Baseline: PR #549 — LeakyReLU² + Legal TTT + Parallel Muon

Source: https://github.com/openai/parameter-golf/pull/549

## 3-Seed Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | Post-TTT bpb | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|--------------|----------|----------|----------|
| 1337 | 83.3ms | 7,179 | 1.1217 | **1.1192** | -0.0025 | 410s | 15,977,386 |
| 42 | 83.4ms | 7,182 | 1.1227 | **1.1200** | -0.0027 | 408s | 15,876,510 |
| 2025 | 83.4ms | 7,193 | 1.1212 | **1.1189** | -0.0023 | 408s | 15,990,006 |
| **Mean** | **83.4ms** | **7,185** | **1.1218** | **1.1194 (std 0.0006)** | **-0.0025** | **~409s** | |

## Key Numbers

- **val_bpb = 1.1194** (3-seed mean, std 0.0006)
- **Pre-TTT bpb = 1.1218** (3-seed mean)
- **Artifact size: ~15.95 MB**
- **Training: ~83.4ms/step, ~7,185 steps**
- **TTT: ~409s, -0.0025 BPB gain**

## Architecture

- 11L, 512d, 8H/4KV, LeakyReLU(0.5)² MLP 3×
- BigramHash(1536), XSA4, Partial RoPE, LN Scale, VE128
- EMA(0.997) + Tight SWA, GPTQ-lite int6 + lzma
- Parameter Banking + Parallel Muon (83.4ms/step)

## Data Order (current baseline)

Default ordering as in train_gpt.py from PR #549. This is what we are experimenting against.
