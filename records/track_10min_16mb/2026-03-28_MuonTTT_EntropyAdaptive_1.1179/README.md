# Record: Muon TTT + Entropy-Adaptive Epochs — val_bpb 1.1179 (3-seed mean)

## 3-Seed Results

| Seed | val_bpb | Eval time | Artifact |
|------|---------|-----------|----------|
| 1337 | **1.1173** | 594s | 15.95MB |
| 42 | **1.1181** | 598s | 16.06MB |
| 2025 | **1.1183** | 603s | 15.94MB |
| **Mean** | **1.1179** | | |
| **Std** | **0.0005** | | |

## Method

### Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3.0x with LeakyReLU(0.5)^2
- XSA on last 4 layers, partial RoPE (16 dims)
- BigramHash embeddings, SmearGate
- Value embeddings on layers 9-10
- Tied embeddings, logit softcap=30

### Training
- Muon optimizer (lr=0.025, momentum 0.99 warmed from 0.92)
- 786K batch tokens, 2048 seq len, 600s wallclock
- EMA (0.997) + SWA every 50 steps
- CROWN-Q QAT during warmdown
- Int6 GPTQ + LZMA compression, 4% pruning

### Test-Time Training (Legal, Score-First)
- Muon-style Newton-Schulz orthogonalized TTT updates
- Entropy-adaptive epoch selection (harder chunks get more epochs)
- 32K token chunks, stride 64
- All blocks unfrozen during TTT
- Score-first: tokens scored BEFORE any weight update

## Credits
- Based on PR #999 by @contributor (Muon TTT + entropy-adaptive epochs)
- CROWN-Q from PR #995 architecture stack
- Built and validated by @TimPietrusky with Claude Code
