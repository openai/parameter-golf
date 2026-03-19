# FP16 Tied Embedding + 10-Layer + Muon WD + NTK Eval + Overtone Init

**Mean val_bpb: 1.2008** (3 seeds, p<0.001)

## Key Techniques

1. **FP16 tied embedding export**: The tied embedding matrix serves as both input lookup AND output projection. Int8 quantization introduces errors that compound in both paths. Keeping `tok_emb.weight` in fp16 during export eliminates most quantization damage at a cost of ~500KB.

2. **10 transformer layers** (up from baseline 9): Muon weight decay reduces artifact size enough to fit an extra layer under 16MB.

3. **Decoupled weight decay for Muon optimizer** (0.02): Muon has no built-in weight decay. Adding `p.mul_(1 - wd * lr)` to all matrix params improves generalization and quantization robustness.

4. **Overtone spectral embedding init**: SVD-reshape the tied embedding matrix to follow a power-law singular value spectrum (`S_k ~ k^{-0.5}`).

5. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization.

6. **NTK-aware RoPE scaling for extended eval**: Train at 1024, evaluate at 2048 with dynamically scaled RoPE.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 2.0283 | 1.2012 | 11044 | 54.33 |
| 42 | 2.0275 | 1.2008 | 10605 | 56.57 |
| 7 | 2.0266 | 1.2003 | 10640 | 56.40 |
| **Mean** | **2.0274** | **1.2008** | | |

Artifact size: ~15.4 MB (under 16 MB limit)
