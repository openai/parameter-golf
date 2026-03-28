# ALL-IN-ONE MONSTER v1 — SOTA Fusion

**Target: sub-1.11 BPB** | 8xH100 SXM, 600s | **Maximum Aggression Mode**

## Combined Techniques (Everything Good From Leaderboard)

### Core Architecture (from our base + top entries)
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- **MLP 3x with LeakyReLU(0.5)²** (current #1 trick -0.003 BPB)
- **Partial RoPE (16/64 dims)** from 1.1248 entry
- **LN Scale 1/sqrt(layer_idx+1)**
- SmearGate + BigramHash(2048) + U-Net skips
- Int6 QAT + zstd-22 compression
- RoPE base=50K + logit softcap=30.0

### Training & Post-Training
- LAWA-EMA + Curriculum (short→long context)
- **GPTQ-lite** per-row optimal clip percentile search
- **Legal Score-First TTT** (full model SGD on validation chunks)
- Muon optimizer + WD=0.04 + warmdown tuning

### Expected Gains
- LeakyReLU²: -0.003
- PartialRoPE + LN Scale: -0.002
- GPTQ-lite + better EMA: -0.001
- Stronger TTT: -0.005 to -0.015
- **Total projected: 1.105 ~ 1.115 BPB**

**This is the final boss submission.**
