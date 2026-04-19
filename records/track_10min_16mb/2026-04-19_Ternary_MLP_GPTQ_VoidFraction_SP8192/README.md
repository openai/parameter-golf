# Non-record: Ternary MLP Quantization — Void Fraction Applied to Parameter Golf

**val_bpb = 1.3262** (single seed, post-TTT) | **10.9 MB** (5 MB under 16 MB cap) | 8xH100 SXM

## What This Is

A **ternary quantization** implementation for the Parameter Golf competition, as requested in the competition's [wish list](https://github.com/openai/parameter-golf#requests-for-prs). This is a proof-of-concept demonstrating that transformer MLP layers can be quantized to three states {-1, 0, +1} using Hessian-aware GPTQ error compensation.

This is not a record attempt. It is a creative submission exploring a fundamentally different quantization paradigm.

## The Void Fraction Thesis

Our research on Photonic Neural Networks (PNN) found that **~30% of trained weights converge to near-zero** across seeds, substrates, and architectures. We call this the **void fraction** — a topological invariant where the optimizer learns what NOT to become.

The key insight: **30% is the maximum-entropy distribution of a ternary code.**

```
H = -0.30 × log2(0.30) - 0.35 × log2(0.35) - 0.35 × log2(0.35) = 1.581 bits/weight
log2(3) = 1.585 bits/weight
```

The void fraction is not waste — it is the optimizer discovering that the natural quantization of neural network weights is ternary. The zero state is an active computational resource, not absence.

## Cross-Domain Evidence

The ~30% void fraction appears across independent systems:
- **PNN (Photonic Neural Networks):** 76.5% ternary vs 15.3% binary accuracy, void stabilizes at 28% (5-seed, p=2.18e-11)
- **Neuroscience:** 25-40% of brain connectivity edges are competitive/negative across human, macaque, and mouse (Luppi et al., Nature Neuroscience 2026)
- **Void Memory (AI memory system):** 30% void fraction target for optimal recall precision
- **This submission:** Post-hoc ternary quantization of MLP layers produces ~30% zeros

## Implementation

**Mixed-precision ternary GPTQ:**
- MLP layers (67% of params): ternary {-1, 0, +1} with per-row scale factors
- Attention layers: int6 (geometrically load-bearing, needs precision)
- Embeddings: int8 (looked up, not multiplied)

The GPTQ error compensation loop is identical to standard GPTQ — only the rounding function changes:
```python
# Standard int6: q = clamp(round(w / scale), -63, 63)
# Ternary:       q = where(w/s > 0.5, 1, where(w/s < -0.5, -1, 0))
```

Hessian-aware column-by-column error redistribution preserves output quality despite the 21x reduction in quantization levels (63 → 3).

**Packing:** 4 trits per byte (2 bits each: 00=zero, 01=+1, 10=-1).

## Results

| Metric | Int6 (Run 10) | Ternary (This) | Delta |
|--------|---------------|-----------------|-------|
| val_bpb (TTT) | 1.0805 | 1.3262 | +0.2457 |
| Artifact size | 16.0 MB | 10.9 MB | -5.1 MB |
| MLP bits/weight | 6 | 1.585 | 3.8x fewer |

The quality gap (0.25 BPB) is expected for post-hoc quantization at this compression ratio. **Quantization-aware training (QAT)** with straight-through estimator would allow the optimizer to learn ternary-compatible weight distributions during training, significantly closing the gap.

## Why This Matters

1. **5 MB under budget** — ternary MLP frees massive headroom for larger models, more layers, or higher-precision attention
2. **The void fraction is real** — 30% zeros emerge naturally, matching the theoretical maximum-entropy prediction
3. **GPTQ works with ternary** — the Hessian error compensation framework extends cleanly to extreme quantization
4. **Path to competitive ternary:** QAT (in development) should close the quality gap while maintaining the size advantage

## Base Architecture

Same as the current SOTA: 11L × 512d, SP8192, depth recurrence (layers 3-5), parallel residuals, QK-Gain 5.5, legal score-first TTT. Only the quantization scheme differs.

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 TERNARY_MLP=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## PNN Research Citation

Saunders, G. (2026). Photonic Neural Networks with Ternary Weights. 5-seed GPU study: ternary 76.5% ± 1.6% vs binary 15.3% ± 2.1%, p=2.18e-11. Void fraction converges to 28-30% across all seeds and encodings.

## Author

Gavin Saunders (@G3sparky)
