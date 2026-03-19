# Muon Weight Decay + NTK Eval + Overtone Init

**Mean val_bpb: 1.2094** (3 seeds, p=0.000213)

## Key Techniques

1. **Decoupled weight decay for Muon optimizer** (0.01): The Muon optimizer has no built-in weight decay. Adding decoupled weight decay (`p.mul_(1 - wd * lr)` after each step) to all matrix-shaped parameters improves both pre-quant generalization (-0.0038 BPB) and reduces the quantization gap (-0.0019 BPB).

2. **Overtone spectral embedding init**: SVD-reshape the tied embedding matrix at init to follow a power-law singular value spectrum (`S_k ~ k^{-0.5}`), giving the model a head start on learning natural language token distributions.

3. **Phase-transition residual mixing**: Initialize the `resid_mix` parameters with a sigmoid schedule so early layers attend more to raw embeddings (x0) while late layers rely on evolved residual representations.

4. **NTK-aware RoPE scaling for extended eval**: Train at `seq_len=1024` but evaluate at `seq_len=2048` using dynamically scaled RoPE base frequencies (`base_adj = base * (scale^(dim/(dim-2)))`), extending the model's effective context at zero parameter cost.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 2.0425 | 1.2097 | 13161 | 45.60 |
| 42 | 2.0425 | 1.2097 | 13224 | 45.37 |
| 7 | 2.0409 | 1.2087 | 13273 | 45.21 |
| **Mean** | **2.0420** | **1.2094** | | |

Artifact size: 14.2 MB (well under 16 MB limit)
