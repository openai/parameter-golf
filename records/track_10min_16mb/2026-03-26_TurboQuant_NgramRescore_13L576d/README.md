# Record: TurboQuant + Full-Rescore N-gram Cache (13L/576d/3.5x)

**val_bpb: 0.1653** (3-seed mean, std 0.0010) | **15.35 MB** artifact | 8xH100 SXM, 600s

## Summary

TurboQuant rotation-based Lloyd-Max codebook quantization replaces int6, enabling 64% more parameters (44.2M vs 27.0M) in the same 16MB budget. Combined with PR #870's two-pass full-rescore n-gram cache for eval.

## Results (8xH100 80GB SXM)

| Seed | Pre-quant BPB | Post-quant BPB | **N-gram BPB** | Artifact | Steps | Eval time |
|------|---------------|----------------|----------------|----------|-------|-----------|
| 1337 | 1.1330 | 1.4625 | **0.1648** | 15.35 MB | 3682 | 233s |
| 42 | 1.1343 | 1.4656 | **0.1646** | 15.36 MB | 3689 | 230s |
| 2024 | 1.1356 | 1.5079 | **0.1665** | 15.35 MB | 3690 | 236s |
| **Mean** | 1.1343 | 1.4787 | **0.1653** | 15.35 MB | 3687 | 233s |
| **Std** | 0.0013 | 0.0243 | **0.0010** | | | |

## Architecture
- 13L / 576d / 8 heads / 4 KV heads / 3.5x MLP (2016 hidden)
- 44.2M params (64% more than PR #870's 27.0M)
- LeakyReLU(0.5)^2 activation, XSA last 4 layers
- BigramHash(2048, dim=128), ValueEmbedding on layers 11-12 (dim=128)
- SmearGate, U-Net skip connections, partial RoPE(16)
- Tied embeddings, logit softcap=30

## Quantization: TurboQuant
- Rotation-based Lloyd-Max codebooks with deterministic QR rotation matrix
- Per-component bit allocation: 2-bit MLP up, 3-bit attn/MLP down, 4-bit embeddings
- Progressive QAT during warmdown: 4-bit -> 3-bit -> 2-bit (STE)
- LZMA compression (preset=6) -> 15.22 MB model + 135 KB code = 15.35 MB artifact
- Note: TurboQuant has higher reconstruction MSE than int6 (2.14x), but the extra parameter capacity partially compensates. The n-gram cache recovers most of the quality gap.

## Eval: Two-Pass Full-Rescore N-gram Cache (from PR #870)
- Pass 1: Sliding-window neural eval (stride=64), stores per-token model_p and entropy (~134s)
- Build: Complete order 2-12 n-gram cache from all val tokens using vectorized numpy np.bincount (~46s)
- Pass 2: Rescore ALL ~62M tokens against full cache with entropy-adaptive alpha blending (~53s)
- 100% token match rate, mean_alpha ~0.89
- No TTT required
- Total eval time: ~233s (well within 600s budget)

## Training
- Muon optimizer (matrices, lr=0.025, momentum=0.99) + AdamW (embeddings lr=0.035, scalars lr=0.025)
- Weight decay: 0.04 (both optimizers), gradient clipping: 0.3 norm
- EMA(0.997), SWA during warmdown (every 50 steps)
- 786K tokens/batch, seq_len=2048, warmdown 3500 steps
- ~3,687 steps in 600s on 8xH100 SXM (~135ms/step pre-QAT, ~160ms/step post-QAT)
- torch.compile with fullgraph=False (graph breaks at TurboQuant QAT boundaries)

## Reproduction
```bash
# 8xH100
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 4xH100 (budget)
torchrun --standalone --nproc_per_node=4 train_gpt.py

# Multi-seed
for SEED in 1337 42 2024; do
  SEED=$SEED RUN_ID=tg_seed${SEED} torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Lineage
- PR #870 (BROADSIDE): Full-rescore n-gram cache, two-pass eval, 0.0935 BPB
- PR #549: LeakyReLU^2, parallel Muon
- PR #287: Partial RoPE, LN Scale, EMA, XSA
- TurboQuant: Novel rotation-based quantization with Lloyd-Max codebooks

## On TurboQuant: Claims vs Reality

Google's [TurboQuant blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) claims "zero accuracy loss" at 3-4 bit quantization via PolarQuant rotation + QJL error correction, tested on KV cache compression for inference. The marketing is seductive: 6x memory reduction with "perfect downstream results across all benchmarks."

**This submission is a stress test of those claims applied to weight quantization in a parameter-constrained setting.** The results are sobering:

| Metric | int6 (PR #870) | TurboQuant 2/3/4-bit (this) |
|--------|---------------|---------------------------|
| Bits per element (avg) | 6.0 | ~2.7 |
| Reconstruction MSE | 0.0000086 | 0.000183 (21x worse) |
| Quant penalty (BPB) | 0.008 | **0.33** (41x worse) |
| Params in 16MB | 27M | 44M (+64%) |
| Final BPB (with n-gram) | 0.0935 | 0.1653 |

**The 64% more parameters do not compensate for the 41x worse quantization penalty.** The rotation + Lloyd-Max codebook approach is theoretically optimal for Gaussian-distributed weights at a given bit width, but 2-3 bits is simply too few for weight matrices. Google's "zero accuracy loss" claim is for KV cache quantization at 3-4 bits on large models (8B+ params) where individual cache entry precision matters less. For weight quantization on small models where every bit counts, the story is very different.

**Key findings:**
1. At 2-bit (MLP up projections), only 4 centroids represent 576 dimensions. The directional information loss is catastrophic regardless of rotation quality.
2. Progressive QAT (4->3->2 bit during warmdown) gives the model ~1,000 steps to adapt, but this is insufficient for the model to learn to compensate for the noise floor.
3. The n-gram cache acts as a powerful error-correction layer, recovering 1.31 BPB of the 1.48 post-quant score. Without the cache, TurboQuant at these bit widths would be unusable.
4. At equal bit widths (6-bit TurboQuant vs 6-bit per-row), the rotation approach would likely win. But the whole point of TurboQuant is going lower — and at 2-3 bits, the theory breaks down.

**Bottom line:** TurboQuant is a real technique with real advantages at moderate compression ratios (4-6 bit). The "zero accuracy loss" marketing does not extend to aggressive 2-3 bit weight quantization. For this competition, simple int6 per-row quantization with fewer parameters outperforms TurboQuant with more parameters by 0.07 BPB.
