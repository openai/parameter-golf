# MHALM V2 — Multi-Head Atlas Language Model

## Summary

MHALM V2 is a geometric language model that replaces the transformer's attention + MLP blocks with kernel-based readout on learned Stäckel coordinates. Three independent encoders map token embeddings to 160-dim coordinate spaces where the metric is approximately diagonal, enabling separable kernel evaluation. Five kernel heads (Nyström, Gabor, Laplacian, Tucker GL, Linear) compute spatial readout in parallel with a 2-pass FFT SSM + causal attention temporal path. The model prefers the spatial (geometric) path over the temporal one when given the choice (learned γ ≈ 1.33).

For the full writeup, see [MHALM: Attention Through Geometry](https://quemy.info/2026-04-29-mhalm-v3-lessons.html).

### Key result

| Metric | Value |
|--------|-------|
| **Competition bpb** | **1.3477** (best) / **1.3481 ± 0.004** (mean 3 seeds) |
| Val loss (nats) | 2.2755 |
| Artifact size | 13.0 MB / 16 MB |
| Stored params | 18.3M |
| Training steps | ~6,000 |
| Training time | 585s on 8×H100 |
| Step time (compiled) | 97.5 ms/step |

### Improvement over V1

| | V1 | V2 | Δ |
|---|---|---|---|
| bpb | 1.4574 | 1.3477 | **−0.107** |
| Artifact | 10.8 MB | 13.0 MB | +2.2 MB |
| Params | 13.6M | 18.3M | +4.7M |

## Architecture

```
tokens → Embedding(1024, 512) + BigramHash(16384×160) + SmearGate
       → RMSNorm
       → HybridAtlasBlock × 2 (U-Net skip connection)
       → W_out → logits (soft_cap 30.0)
```

### Inside each HybridAtlasBlock

**Three Stiefel-enforced chart encoders** (512→160 each, 4-layer MLP, SiLU, tanh output with learnable temperature). Each feeds a dedicated kernel head:

- Ψ₀ → Nyström (Gegenbauer polynomial, causal, R=256)
- Ψ₁ → Gabor (Gaussian × cosine, R=256)
- Ψ₂ → Laplacian (RBF mixture: Gaussian + Laplacian + Matérn-3/2, R=256)
- Tucker GL = Ψ₁ × Ψ₂ (element-wise product, no extra params)
- Linear = raw Ψ₀ output (d=160)

**Two parallel paths from encoder outputs:**

1. **Temporal path** — z_cat = [z₀, z₁, z₂] ∈ ℝ⁴⁸⁰ feeds 2-pass (SSM ∥ Attention):
   - FFT SSM (S4D-family, cuFFT causal conv, Weyl spectral init, O(T log T))
   - 8-head causal self-attention (RoPE, XSA), 2 layers per pass
   - Gated combination: σ(g)·SSM + (1-σ)·Attn
   - Pass 2 refines residual from Pass 1 (independent weights)

2. **Spatial path** — each encoder feeds its kernel head → stacked GEMM → mixed ∈ ℝ¹⁰²⁴

**Output:** logits = W_out(H_temporal) · Eᵀ + γ × mixed (γ init=0, learned to ~1.33)

### Key V1→V2 changes

| Change | Impact (est_bpb) |
|--------|:------:|
| Stiefel enforcement fix (power iteration) | −0.065 |
| Weyl spectral SSM init | −0.062 |
| d_max 128 → 160 | −0.036 |
| z-space temporal processing | −0.034 |
| FFT SSM (cuFFT) | −0.031 |
| Surgical Muon routing | −0.022 |

### Training

- Decoupled optimizer: Muon (2D matrices) + AdamW (scalars)
- Stäckel penalty (β=0.02, soft diagonal covariance)
- SWA (last 40% of training)
- Whole-model torch.compile + 3-step warmup
- Int8 quantisation + zstd-22 compression

## Running

```bash
# Train on 8×H100 (golf submission)
torchrun --standalone --nproc_per_node=8 train_gpt.py \
    --mode golf --stiefel --stiefel-scale \
    --n-encoder-hidden 4 --d-max 160 \
    --data-dir ../../data/datasets/fineweb10B_sp1024/ \
    --tokenizer-path ../../data/tokenizers/fineweb_1024_bpe.model

# Quick smoke test
python train_gpt.py --mode smoke --stiefel --stiefel-scale
```
