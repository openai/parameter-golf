# Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init

**Mean val_bpb: 1.1748** (3 seeds, p<0.001)

## Key Techniques

1. **Sliding window evaluation** (stride=64, seq_len=1024): Every token scored with 960+ context instead of 0-1023 average. Compiled `forward_logits` method for efficient batch inference.

2. **FP16 tied embedding export**: Keep `tok_emb.weight` in fp16 — int8 errors compound through both input and output paths.

3. **10 transformer layers** (up from 9): Muon weight decay compresses enough to fit the extra layer.

4. **Decoupled weight decay for Muon optimizer** (0.02): Improves generalization and quantization robustness.

5. **Overtone spectral embedding init**: SVD power-law spectrum shaping (`S_k ~ k^{-0.5}`).

6. **Phase-transition residual mixing**: Sigmoid-scheduled `resid_mix` initialization.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.9849 | 1.1756 | 10424 | 57.55 |
| 42 | 1.9827 | 1.1742 | 10710 | 56.06 |
| 7 | 1.9830 | 1.1744 | 10498 | 57.18 |
| **Mean** | **1.9835** | **1.1748** | | |

Artifact: ~14.7 MB | Eval time: ~162s (sliding window)
