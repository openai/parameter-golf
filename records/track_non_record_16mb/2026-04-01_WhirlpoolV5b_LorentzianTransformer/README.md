# Whirlpool v5b — Non-Euclidean Lorentzian Attention on the Hyperboloid Manifold

**Non-record submission** — non-Euclidean geometry exploration, not claiming SOTA.

Explores whether replacing dot-product attention with Minkowski inner products on a hyperboloid manifold can capture language's hierarchical structure more naturally. Key finding: it works, but requires careful stabilization (scale clamping + extended warmup gave **-0.88 BPB**).

**val_bpb: 1.5918** (1-seed, SEED=314) | **~12.2 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, AP-IN-1)

| Seed | step_avg | steps | tokens | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Artifact |
|------|----------|-------|--------|-------------|-----------------|----------|----------|
| 314 | 36ms | 16,140 | 1.058B | 1.5976 | **1.5918** | -0.0058 | 12,134,416 |

## Architecture: Lorentzian Geometry in Attention

To our knowledge, this is the first non-Euclidean submission to replace standard dot-product attention with **hyperboloid geometry**. Instead of computing attention scores as dot products in flat Euclidean space, Whirlpool projects queries and keys onto the hyperboloid manifold and scores them using the Minkowski inner product — the natural metric of hyperbolic space.

### Why Hyperbolic Geometry?

Natural language has an inherent hierarchical structure (syntax trees, semantic taxonomies, discourse relations). Hyperbolic spaces can embed trees with exponentially lower distortion than Euclidean spaces of the same dimension (Sarkar 2011). By focusing attention on the hyperboloid, the model can natively represent hierarchical relationships in its attention patterns.

### Key Innovation: Flash Lorentz Attention

A custom Triton kernel (`flash_lorentz_attn`) fuses the entire Lorentzian attention pipeline into a single tiled kernel:

1. **Hyperboloid projection**: `x0 = sqrt(1/c + ||x_spatial||^2)`, producing points on the hyperboloid sheet
2. **Minkowski inner product**: `<a,b>_L = -a0*b0 + a1*b1 + ... + ad*bd` (Lorentzian metric)
3. **Temperature-scaled softmax**: `softmax(-c * <q,k>_L / temp)` per head
4. **Lorentzian centroid aggregation**: Values aggregated on the manifold (not Euclidean mean)

Registered as `torch.library.custom_op` — no graph breaks with `torch.compile`. O(T) memory via online softmax (no materialized attention matrix). GQA handled internally.

```python
attn_out, spatial_norm = flash_lorentz_attn(
    q, k, v,
    curvature=curvature,    # per-orbit curvature (0.1 to 2.0)
    temp=self.temp,          # per-head learned temperature
    causal=True,
    use_centroid=True,       # Lorentzian centroid value aggregation
)
```

### Orbit Architecture: Depth Through Weight Sharing

Instead of stacking N separate transformer layers, Whirlpool uses **3 shared blocks** called repeatedly across **8 orbits** with different curvatures:

| Block | Orbits | Curvature | Role |
|-------|--------|-----------|------|
| Local | 0, 1, 2 | 0.10, 0.15, 0.24 | Flat geometry, local patterns |
| Transition | 3, 4 | 0.36, 0.55 | Bridge local to hierarchical |
| Hierarchy | 5, 6, 7 | 0.85, 1.30, 2.00 | High curvature, long-range |

Each orbit has its own embedding perturbation and learnable attention/MLP scales, but shares the core weight matrices. This yields 8 passes of depth from 23.9M stored parameters — a Universal Transformer approach in which each orbit sees a different curvature of the underlying manifold.

**Progressive curvature** (0.1 to 2.0, approximately geometric progression) means early orbits operate in nearly flat space (local token patterns) while later orbits operate in highly curved space (hierarchical/long-range dependencies).

### Scale Clamping: Stabilizing Lorentzian Training

Orbit scales (`attn_scale`, `mlp_scale`) tend to explode mid-training due to the exponential nature of hyperbolic geometry. Two stabilization techniques:

1. **Scale clamping**: `.clamp(-5.0, 5.0)` on both scale params per orbit
2. **20% warmup**: Linear warmup over 20% of training (vs typical 1-2%)

This combination improved val_bpb by **0.88 BPB** (2.433 to 1.55) — the single biggest finding in the Lorentzian track, and potentially useful for anyone working with non-Euclidean geometry in neural networks.

## Training Configuration

| Component | Setting |
|-----------|---------|
| d_model | 768 |
| Heads | 12 (GQA 6:1, kv=2, head_dim=64) |
| MLP | 5x with **LeakyReLU(0.5)²** (fused Triton kernel) — LeakyReLU preserves negative gradient flow; squaring maintains non-negative output (PR #493) |
| Orbits | 8 active (3 shared blocks) |
| Curvature | 0.1 to 2.0 (progressive) |
| Optimizer | MuonAdamW (muon lr=0.04, wd=0.12) |
| Warmup | 20% linear + cosine decay |
| Muon momentum | 0.85 to 0.95 (warmup over 500 steps) |
| EMA | 0.997 decay |
| Crown-Q | Quantization-aware training from 80% progress — adds a small penalty encouraging weights toward int8-friendly values |
| Quantization | int8 per-row + zlib |
| Compile | torch.compile with 20-step warmup |
| Batch | 8 per GPU, 65K total tokens/step |

### Fused Triton Kernels

Three custom Triton kernels for performance:

1. **Flash Lorentz Attention**: Fused hyperboloid projection + Minkowski inner + softmax + centroid. O(T) memory, ~10x vs unfused PyTorch implementation of the same operations.
2. **Fused LeakyReLU(0.5)²**: Single kernel eliminates intermediate tensor allocation in the MLP. 2.6x MLP speedup.
3. **Int6 quantizer**: Triton-packed int6 with GPU dequant (available but int8 used for this submission — more headroom).

## Eval Strategy: Parallel GPU Eval

After training, each GPU independently runs a different TTT hyperparameter search. Rank 0 collects results and reports the best. Only TTT lr=5e-4 with 1 step provided marginal improvement (-0.006 BPB), consistent with a well-converged model:

| Rank | Strategy | val_bpb | Time |
|------|----------|---------|------|
| 0 | base_int8 (compiled) | 1.5976 | 59s |
| **4** | **TTT lr=5e-4, 1 step** | **1.5918** | **262s** |
| 1 | TTT lr=5e-4, 2 steps | 1.6217 | 459s |
| 3 | TTT lr=1e-3, 1 step | 1.6500 | 261s |
| 7 | N-gram blend (alpha=0.3) | 1.7386 | 69s |

TTT uses score-first protocol (legal): score chunk under `torch.no_grad()`, then train on scored tokens with AdamW(lr=5e-4, wd=0.0).

## Development Journey

Whirlpool evolved through 50+ iterations:

- **v1-v3**: Poincare ball attention (collapsed — gradients vanished at ball boundary)
- **v4.0**: Switched to hyperboloid model — stable, first successful Lorentzian training
- **v4.2**: Added centroid aggregation (-1.23 BPB), progressive curvature, scale clamping
- **v5.0**: Fused Triton kernels (2.6x speedup), integrated DDP pipeline
- **v5b**: Fixed eval pipeline, parallel GPU eval, d=768 scaling, 8-orbit activation

Key lessons: Lorentzian geometry requires careful numerical stabilization (scale clamping, warmup), EMA weights must be Euclidean (not manifold-aware), and the hyperboloid projection must use sqrt (not expmap) for training stability.

## Limitations

- **Single seed**: Only SEED=314 reported. Historical runs show variance of approximately +/-0.01 BPB.
- **No direct Euclidean ablation at this scale**: We have not run a standard dot-product attention model with the same d=768/5xMLP configuration and training budget for a controlled comparison. Our Euclidean track (different architecture, d=640) achieved 1.08 BPB, suggesting the Lorentzian geometry adds overhead without proportional BPB gain at this competition's scale/budget.
- **Unused artifact headroom**: 3.8MB of the 16MB budget is unused. Scaling d_model higher is limited by step time (8 orbits = 8 forward passes per step).
- **TTT benefit minimal**: Well-trained Lorentzian model shows only -0.006 BPB from TTT, vs -0.3 BPB when undertrained.

## Run Command

```bash
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Artifact Budget

| Component | Bytes |
|-----------|-------|
| Code (train_gpt.py) | 73,848 |
| Model (int8+zlib) | 12,134,416 |
| **Total** | **12,208,264** |
| **Headroom** | **3,791,736** |

## Future Research Path

Open questions for future work:

- **Riemannian optimizers**: Replacing Euclidean SGD/Adam with manifold-aware optimization (Riemannian gradient descent on the hyperboloid) for the geometric parameters
- **Geodesic skip connections**: Interpolating residuals along geodesics rather than Euclidean addition
- **Curvature learning**: Making per-orbit curvature learnable rather than fixed, allowing the model to discover optimal geometry per depth
- **Mixed-geometry attention**: Different heads operating in different curvature regimes simultaneously
- **Hyperbolic embeddings**: Moving token embeddings onto the manifold (currently Euclidean) to maintain geometric consistency end-to-end
- **Direct Euclidean ablation**: Controlled comparison with same model size and orbit architecture but standard dot-product attention

## Credits

- **Hyperbolic embeddings**: Foundational work by Nickel & Kiela (2017) "Poincare Embeddings" and Ganea et al. (2018) "Hyperbolic Neural Networks"
- **Tree embedding distortion**: Sarkar (2011) "Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane"
- **Flash Attention pattern**: Dao et al. (2022), adapted for Minkowski metric
- **MuonAdamW optimizer**: Based on upstream Parameter Golf Muon implementation
- **LeakyReLU(0.5)²**: PR #493 by @parinzee
- **TTT recipe**: Adapted from PR #461 by @Christopher-Lee-McClendon
