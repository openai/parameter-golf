# FP16 Embed + 10-Layer + Muon WD + NTK@1408 + Overtone Init

**Mean val_bpb: 1.2000** (3 seeds, p<0.001)

## Key Techniques

1. **FP16 tied embedding export**: Keep `tok_emb.weight` in fp16 instead of int8 during export. The tied embedding serves as both input lookup AND output projection — int8 quantization errors compound in both paths.

2. **10 transformer layers** (up from 9): Muon weight decay compresses the artifact enough to fit an extra layer under 16MB.

3. **Decoupled weight decay for Muon optimizer** (0.02): Muon has no built-in regularization. Adding `p.mul_(1 - wd * lr)` to matrix params improves generalization and quantization.

4. **NTK-aware RoPE eval at 1.375x** (eval@1408, train@1024): Dynamic RoPE base frequency scaling at 1.375x training length is optimal — less frequency distortion than 2x while still extending effective context.

5. **Overtone spectral embedding init**: SVD power-law spectrum shaping (`S_k ~ k^{-0.5}`).

6. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 2.0268 | 1.2004 | 10562 | 56.81 |
| 42 | 2.0260 | 1.1999 | 10525 | 57.04 |
| 7 | 2.0254 | 1.1996 | 10710 | 56.02 |
| **Mean** | **2.0261** | **1.2000** | | |

Artifact: ~14.6 MB (under 16 MB limit)
