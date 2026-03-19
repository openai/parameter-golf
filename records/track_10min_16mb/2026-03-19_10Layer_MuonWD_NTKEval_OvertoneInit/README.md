# 10-Layer + Muon Weight Decay + NTK Eval + Overtone Init

**Mean val_bpb: 1.2029** (3 seeds, p=0.000560)

## Key Techniques

1. **10 transformer layers** (up from baseline 9): Muon weight decay reduces artifact size enough to fit an extra layer under 16MB. The extra model capacity (19M → from 17M params) more than compensates for 24% fewer training steps.

2. **Decoupled weight decay for Muon optimizer** (0.02): Muon has no built-in weight decay. Adding decoupled weight decay (`p.mul_(1 - wd * lr)`) to all matrix-shaped parameters improves both generalization and quantization robustness.

3. **Overtone spectral embedding init**: SVD-reshape the tied embedding matrix at init to follow a power-law singular value spectrum (`S_k ~ k^{-0.5}`), matching natural language statistics.

4. **Phase-transition residual mixing**: Initialize `resid_mix` with a sigmoid schedule so early layers attend more to raw embeddings while late layers rely on evolved representations.

5. **NTK-aware RoPE scaling for extended eval**: Train at `seq_len=1024`, evaluate at `seq_len=2048` using dynamically scaled RoPE base frequencies.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 2.0312 | 1.2030 | 10605 | 56.58 |
| 42 | 2.0288 | 1.2016 | 10606 | 56.57 |
| 7 | 2.0330 | 1.2041 | 10605 | 56.49 |
| **Mean** | **2.0310** | **1.2029** | | |

Artifact size: 14.3-14.4 MB (under 16 MB limit)
