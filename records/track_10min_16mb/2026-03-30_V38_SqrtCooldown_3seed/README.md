# V38: CROWN-Q + Full GPTQ + Legal TTT + Sqrt Cooldown

**val_bpb: 1.1174 (3-seed mean, std 0.0004)**

## Results

| Seed | EMA BPB | TTT BPB | Artifact |
|------|---------|---------|----------|
| 1337 | 1.1376 | **1.1170** | 15,961,751 |
| 42 | 1.1381 | **1.1176** | 15,850,151 |
| 7 | 1.1383 | **1.1176** | 15,844,080 |
| **Mean** | 1.1380 | **1.1174** | |
| **Std** | | 0.0004 | |

## Architecture

- 11L, 512d, GQA 8/4, MLP 3x relu²
- XSA on all 11 layers
- BigramHash 2048, Shared VE128 (layers 9,10)
- Partial RoPE 16/64, LN Scale, SmearGate + OrthoInit
- 26.99M parameters

## Training

- Muon optimizer (lr=0.025, wd=0.04)
- 600s wallclock on 8xH100 SXM, 89ms/step (FA3 Hopper)
- ~6650 steps, coprime multi-shard data loader
- EMA 0.997, SWA (last ~150 steps)
- Late QAT at 50% warmdown
- CROWN-Q: curvature-weighted quantization penalty during warmdown
- **Sqrt cooldown schedule**: holds learning rate higher longer during warmdown

## Quantization & Eval

- Full Cholesky GPTQ with act-order (self-generated calibration, no val data)
- int6 + zstd-22 compression
- Score-first AdamW TTT (lr=0.0001, 3 epochs, last 2 blocks unfrozen)
- Sliding window eval stride=32

## Compliance

- Training: 600s wallclock (hard cap)
- GPTQ calibration: self-generated tokens (no validation data used)
- TTT: legal score-first (each token scored before any gradient update using it)
- All artifacts < 16,000,000 bytes
