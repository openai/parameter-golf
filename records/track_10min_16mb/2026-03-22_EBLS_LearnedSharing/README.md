# EBLS Learned Sharing

**Track**: 10 min / 16 MB
**Val BPB (post-quant)**: 1.3441
**Val BPB (pre-quant)**: 1.2105
**Artifact size**: 16,224,826 bytes
**Date**: 2026-03-22

## Approach

Empirical Bayes Layer Sharing (EBLS): 3 shared transformer blocks, each applied 3× for 9 effective layers. Per-virtual-layer rank-8 LoRA deviations gated by learned shrinkage factors γ_i = σ(logit_i). Shrinkage regularization encourages weight sharing unless deviation helps.

## Architecture

- **Dimension**: 1024, **Heads**: 16Q / 4KV (GQA)
- **Layers**: 3 shared blocks × 3 = 9 virtual layers
- **LoRA rank**: 8 (attention + MLP)
- **MLP**: 3× expansion with ReLU²
- **Features**: SmearGate, BigramHash(10240), U-Net skips
- **Optimizer**: Muon (WD=0.04) + Adam (LoRA, embeddings, scalars)
- **Quantization**: Int6 STE QAT + zstd-22

## Key Finding

The model discovers optimal sharing automatically:
- MLP gammas → 0.0000 across all virtual layers (fully shared)
- Attention gammas → 0.0035 for layer 0, ~0 otherwise (minimal specialization)

## Reproduce

```bash
bash eval/eval.sh
```
