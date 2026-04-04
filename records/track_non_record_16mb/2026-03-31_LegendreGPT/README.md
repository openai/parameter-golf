# LegendreGPT

LegendreGPT generates all transformer layer weights from a small set of Legendre polynomial coefficients, compressing 22 middle layers into 6 coefficient matrices per weight type. As far as I know, this is the first time anyone has applied orthogonal polynomial weight parameterization to transformer language models. The greatest news is that it learns.

## Result

| Metric | Value |
|--------|-------|
| Pre-quantization val_bpb | 1.2079 |
| Post-quantization val_bpb (INT7+zlib) | 1.2353 |
| Post-quantization val_bpb (mixed INT8/INT7+LZMA) | 1.2266 |
| Compressed model size | 15.70 MB |
| Architecture | dim=512, 24L (2 groups), g5/2, GQA 8/4 |
| Training | 60k steps, 80 shards, 1x RTX 5090 (~27h) |

Note: The INT7+zlib number (1.2353) is from the training script's built-in roundtrip validation (see train.log). The mixed INT8/INT7+LZMA number (1.2266) comes from a separate post-hoc quantization where Legendre orders 0-1 use INT8 and the rest use INT7, compressed with LZMA instead of zlib.

## How It Works

Each weight matrix in the transformer is a function of depth:

```
W(layer_l) = sum_{k=0}^{K-1} C_k * P_k(t_l)
```

`P_k` are Legendre polynomials. `t_l` maps layer index to [-1, 1]. `C_k` are learned matrices. With K=6 (degree 5), I specify 11 unique layers from 6 coefficient matrices per weight type — and I have two independent groups of 11.

Think of it like an equalizer: the polynomials are fixed frequencies (constant, linear, quadratic...), and the coefficients are the sliders. Training only adjusts the sliders, never the frequencies.

**Why Legendre and not monomials?** Orthogonality. Monomials (1, t, t^2...) become catastrophically ill-conditioned at higher degrees. Legendre polynomials stay well-behaved. NANODE (Massaroli et al., 2020) showed this matters for Neural ODEs. I confirmed it matters for transformers too.

## Architecture

```
[Factorized Embedding]              <- ALBERT-style, 1024 -> 128 -> 512
[Independent Block 0]               <- own weights
[Legendre Group A: 11 layers]       <- coefficients A (degree 5 attn, 2 FFN)
[Legendre Group B: 11 layers]       <- coefficients B (independent)
[Independent Block 23]              <- own weights
[RMSNorm -> Tied Logit Head]
```

The 2-group split is important. With 1 group, each coefficient affects all 22 layers. If layer 5 wants the coefficient to go up but layer 15 wants it to go down, the gradients cancel and nothing moves. With 2 groups, each coefficient only fights with ~11 layers instead of 22.

Each layer also has cheap independent scalars: attention scale, MLP scale, residual mixing ratio, query gain, and RMSNorm params. These cost < 0.05 MB total and let each layer fine-tune its behavior without defeating the compression.

Other details: GQA (8 heads, 4 KV heads), ReLU^2 MLP at 3x dim, RoPE, logit soft-capping at 30.

## Parameter Budget

| Component | Params | INT8 (MB) |
|-----------|--------|-----------|
| Factorized embedding | 196,608 | 0.20 |
| Sandwich block (first) | 2,361,352 | 2.36 |
| Legendre Group A coefficients | 9,437,250 | 9.44 |
| Legendre Group B coefficients | 9,437,250 | 9.44 |
| Per-layer lightweight params | 45,232 | 0.05 |
| Sandwich block (last) | 2,361,352 | 2.36 |
| **Total** | **23,839,044** | **15.70 MB compressed** |

Compression: mixed precision quantization (INT8 for Legendre orders 0-1, INT7 for orders 2-5 and sandwich blocks) + LZMA.

## Training

- **Muon optimizer** for all 2D weight matrices, Adam for embeddings and scalars
- **Per-order learning rates:** 1.1x higher per polynomial order. Order 0 at 0.025, order 5 at 0.040. Higher orders capture finer detail and need more push.
- **Linear LR decay** from 0.2 to 0.0 over 60k steps
- **Momentum cooldown:** Muon momentum decays from 0.95 to 0.05 over steps 10k-60k. Discovered accidentally when a checkpoint resume zeroed the momentum buffers and the model learned 8x faster. High momentum dampens updates too much near convergence.
- **Batch size:** 393,216 tokens. Legendre coefficients serve 11+ layers and benefit from clean gradient estimates.
- **Data:** 80 FineWeb sp1024 shards (~8B tokens)
- **Hardware:** 1x RTX 5090, ~27 hours total

## Training Curve

| Step | val_bpb | lr_mul |
|------|---------|--------|
| 1,000 | 1.48 | 0.197 |
| 5,000 | 1.35 | 0.183 |
| 10,000 | 1.28 | 0.167 |
| 20,000 | 1.24 | 0.133 |
| 30,000 | 1.22 | 0.100 |
| 40,000 | 1.21 | 0.067 |
| 50,000 | 1.21 | 0.033 |
| 60,000 | 1.2054 | 0.000 |

## What I Learned

**Dimension matters most.** For a fixed byte budget, bumping dim from 512 to 640 gave 0.02 BPB. Raising polynomial degree from g5/2 to g8/4 gave ~0.01. Wider layers beat more variation between them.

**2 groups beat 1 group.** Each Legendre coefficient affects all layers in its group. With 1 group of 22 layers, gradients from different layers partially cancel. Splitting into 2 groups of 11 halves the cancellation and improves convergence at the same parameter cost.

**Wrap doesn't help.** Tested circular weight topology (W = W - round(W)) with standard init, 8x init, and wrap-aware gradient modifications. Smooth weights win consistently — the continuity prior between adjacent layers is correct.

**LoRA is less efficient than higher degree.** Per-layer low-rank corrections (W = W_legendre + A*B) at rank 8 underperform simply raising Legendre degree. g6/3 without LoRA beats g5/2 + LoRA r8 — the bytes are better spent on polynomial expressivity.

**Momentum cooldown helps late training.** High momentum (0.95) dampens updates too much near convergence. Decaying to 0.05 in the second half of training allows finer adjustments when the model is close to a good minimum.

**Larger batches help disproportionately.** Going from 262k to 393k tokens/batch improved convergence visibly.

**Mixed precision quantization is key.** Legendre order 0-1 (constant and linear components) carry most of the weight information and need INT8 precision. Higher orders (finer detail) tolerate INT7. This gives near-INT8 quality at near-INT7 size.

## Experiments

| Config | Steps | val_bpb | Takeaway |
|--------|-------|---------|----------|
| dim=256, 8L, g5/2, 1 shard | 3,000 | 1.67 | Architecture works |
| dim=512, 24L, g5/2, 1 shard | 2,000 | 1.39 | Full model, 8.6 MB |
| dim=640, 24L, g6/3, 80 shards | 30,000 | 1.214 | Best BPB but 18.4 MB |
| dim=576, 24L, g6/3, 80 shards | 30,000 | 1.22 | Fits budget, tight |
| dim=512, 2-group g5/2, wrap | 3,000 | 1.70 | Wrap hurts |
| dim=640, g5/2 + LoRA r8 | 5,000 | 1.30 | LoRA < higher degree |
| **dim=512, 2-group g5/2, 80 shards** | **60,000** | **1.2054** | **Final submission** |

## Related Work

**NANODE** (Massaroli et al., NeurIPS 2020) used Legendre polynomials to parameterize Neural ODE weights for PDE surrogate modeling. LegendreGPT extends this idea to transformer language models.

**ALBERT** (Lan et al., 2020) shares identical weights across all layers. LegendreGPT generalizes this: degree 0 (single constant coefficient) is exactly ALBERT. Higher degrees let layers diverge smoothly.

**Subformer** (Reid et al., 2021) showed that sandwich-style sharing (independent first/last layers, shared middle) works better than uniform sharing. I use the same structure.

## What I'd Try Next

- **2D compression:** Legendre polynomials for the depth axis, DCT for the width axis. Could push dim to 1024+ in 16 MB.
- **Learned basis:** PCA from a pretrained model's weights instead of fixed Legendre. The optimal basis probably isn't polynomial.
- **Low-rank high orders:** Full-rank for orders 0-2, low-rank for orders 3+. More expressivity per byte.
- **Learnable layer positions:** Let the model learn optimal spacing in [-1, 1] instead of uniform.
- **Proper 8xH100 run:** All my runs were on a single RTX 5090. The competition target is 8xH100 in 10 minutes. Larger batch, fewer steps, different schedule.

## Reproducibility

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install sentencepiece huggingface-hub datasets
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Copy train_gpt_legendre.py to parameter-golf/

MODE=full RUN_ID=legendregpt \
  LEGENDRE_GROUPS=2 \
  NUM_VIRTUAL_LAYERS=24 MODEL_DIM=512 \
  LEGENDRE_DEGREE_ATTN=5 LEGENDRE_DEGREE_FFN=2 \
  ITERATIONS=60000 TRAIN_BATCH_TOKENS=393216 \
  MAX_WALLCLOCK_SECONDS=0 LR_SCHEDULE=linear \
  python3 train_gpt_legendre.py
```

## Author

**Sergio Cernuda Cueto**
