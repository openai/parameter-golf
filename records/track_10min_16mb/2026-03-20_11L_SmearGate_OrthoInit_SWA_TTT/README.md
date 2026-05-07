# 11L Int6 QAT + SmearGate + OrthoInit + SWA + TTT + NTK-RoPE

**Author:** yahya010
**Date:** 2026-03-20
**Score:** val_bpb = 1.1478 (seed 1337)

## Summary

Full-stack submission combining 12 techniques:

| Component | Details |
|-----------|---------|
| **Layers** | 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA) |
| **MLP** | 3x expansion (hidden=1536), ReLU squared |
| **Quantization** | STE Int6 QAT (zero quant gap), fp16 tied embeddings |
| **Compression** | zstd-22, artifact 15.76 MB |
| **SmearGate** | Learned sigmoid token blending |
| **BigramHash** | 2048-bucket hash embedding (dim=128) |
| **OrthoInit** | Orthogonal init + muP scaling for output projections |
| **Optimizer** | Muon (WD=0.04, momentum=0.99, LR=0.025) |
| **SWA** | 8 checkpoint average during warmdown (scale < 0.5, every 200 steps) |
| **Position** | NTK-RoPE (base=50000) |
| **TTT** | Full-weight SGD on val data (lr=0.002, 3 epochs, freeze 2 blocks) |
| **Eval** | Sliding window stride=64 |

## Results

| Seed | Steps | Step Avg | Post-Quant BPB | Sliding BPB |
|------|-------|----------|----------------|-------------|
| 1337 | 5,166 | 116ms | 1.1712 | **1.1478** |

- Artifact size: 15,757,600 bytes
- Training: 600s wallclock
- TTT: 73s
- Sliding window eval: ~370s
- Total eval: ~443s (under 600s budget)

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults in the script match the submitted configuration.
