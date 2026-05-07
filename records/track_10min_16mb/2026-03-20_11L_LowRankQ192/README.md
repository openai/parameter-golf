# 11L Low-Rank Q192

## Results

| Seed | Steps | step\_avg | val\_bpb (sliding) | Artifact Size |
|------|-------|----------|--------------------|---------------|
| 1337 | 7732  | 77.6ms   | 1.1548             | 14,747,273    |
| 42   | 7821  | 77.1ms   | 1.1552             | 14,939,593    |
| 113  | 7597  | 79.0ms   | 1.1575             | 14,676,072    |
| **Mean** | — | —     | **1.1558**         |               |

All runs use clean compile cache (`rm -rf ~/.cache/torch/inductor_cache/`), zstd-22 compression.

**vs official record** (SlidingWindow\_FP16Emb\_10L\_MuonWD\_OvertoneInit, val\_bpb=1.1748):
- Improvement: **-0.0190 bpb** / **-0.031 nats**
- One-sided t-test: t≈22, df=2, **p < 0.001**

## How to run

```bash
NUM_LAYERS=11 WEIGHT_DECAY=0.038 SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```

## Key Techniques

1. **Low-rank Q factorization (r=192)**: Q projection factored as `c_q_down(512->192)` + `c_q_up(192->512)`. Q matrices have extreme condition numbers (100M+) and effective rank 89-114 out of 192, confirming rank 192 is sufficient. The factored representation is more int6-quantization-friendly: low-rank structure compresses better, reducing the fp32-to-int6 gap.

2. **11 transformer layers** with encoder-decoder skip connections (5 encoder + 6 decoder), using parameter savings from low-rank Q.

3. **Int6 per-row quantization + zstd-22 compression** for MLP and attention weights. Scalar/control parameters kept in fp32.

4. **Sliding window evaluation** (stride=64) for final score.

## Architecture

- 11 layers, model\_dim=512, 8 attention heads, 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu-squared activation
- Low-rank Q: `c_q_down (512->192)` + `c_q_up (192->512)` per layer
- Tied embeddings, vocab=1024, logit\_softcap=30
- RoPE (base=10000), RMSNorm, residual mixing

## Motivation

Standard Q projection uses a full 512x512 matrix, but the Muon optimizer pushes all singular values toward 1, while Q naturally wants to operate in a lower-rank subspace (effective rank 89-114/192). Factoring Q to rank 192 makes this structure explicit, which has two benefits: (1) the model trains at the same per-step quality, and (2) the factored weights quantize significantly better under int6 per-row quantization because low-rank matrices have higher information density per element.

## Experiments That Didn't Work

The following ideas were explored through weight analysis and experiments but did not improve val\_bpb:

**1. Legendre resid\_mix initialization**

After training, the `resid_mix` parameters (mix0 and mix1) show a clear depth-dependent pattern: mix0 has a Block-0 outlier with a U-shape for deeper layers, and mix1 follows a strong U-shape with negative values in the middle (embedding subtraction). These patterns fit well to 4th-order Legendre polynomials. I tried initializing resid\_mix with integer Legendre coefficients at half-scale (interpolating between standard init and Legendre target) to give the optimizer a warm start. However, experiments showed no measurable improvement — Adam converges the scalar resid\_mix parameters to their targets within ~200 steps regardless of initialization. The Legendre shape is correct but the optimizer doesn't need help finding it.

**2. Content-Dependent Pre-Rotation on Block 0's MLP**

I introduced a content-dependent 2D rotation before each MLP's first linear layer (fully-connected, 512→1536): a small projection (`angle_proj`, 512→32, zero-init) computes 32 rotation angles from the input, then rotates 32 pairs of input dimensions before feeding into the fully-connected layer. This adds SwiGLU-like content-dependent feature mixing at only 1% parameter cost (16K params per layer) and without sacrificing MLP width — unlike SwiGLU which requires a third gate matrix (50% more MLP params, forcing MLP 3x down to 2x in a 16MB budget).

Experiments confirmed the rotation is genuinely useful: the model learned large rotation angles (~146°, not small perturbations), used near-full effective rank (29-30/32 pairs independently active), and achieved higher per-step quality in later training compared to the baseline. Further analysis showed Block 0 learned the strongest rotation (row\_norm=2.55, focusing on raw embedding dimensions 359, 156, 423...) while deeper layers learned weaker rotation (row\_norm~1.1, sharing contextual dimensions 4, 12, 23...), suggesting Block 0 benefits most. However, `torch.compile` generates separate kernels for the rotation operations (cos/sin + concatenation), adding ~9 seconds of fixed compilation overhead — theoretically ~2ms of compute inflated to 9ms by graph-level inefficiency. I also tested a Block-0-only variant to reduce overhead, but the fixed compilation cost remained. In a 600-second budget, this overhead costs ~100 training steps, which negated the per-step quality gain.

This remains a promising direction: content-dependent rotation provides norm-preserving, information-lossless feature mixing (det(R)=1) as a near-free alternative to gating mechanisms in parameter-constrained settings. The bottleneck is purely at the operator compilation level, not the method itself.

**3. Depth-attention residual (AttnRes) architecture**

Inspired by Moonshot AI's [Attention Residuals](https://arxiv.org/abs/2603.15031), I explored replacing the standard residual stream with a depth-attention mechanism: each layer's input is `emb + depth_attn(δ₀..δ_{i-1})` where `depth_attn` uses learned position bias (Legendre polynomials) and content-based routing over all previous layers' outputs. The motivation was (a) selective delta combination for better gradient flow, and (b) quantization error suppression (softmax weights sum to 1, reducing error accumulation).

However, attention residual turns out to be counterproductive in small, dense models like this one. In Kimi-K2's MoE setting, attention residual helps route across sparsely-activated experts. In our dense 512-dim model, it actually suppresses the residual stream: softmax constrains routing weights to be non-negative and sum-to-1, but weight analysis of the baseline's `resid_mix` revealed the optimal depth routing requires negative weights (middle layers subtract embedding with mix1 ≈ -4) and non-normalized weights — patterns that softmax fundamentally cannot express. Additionally, unnormalized Values in depth attention caused block polarization where only 3 of 9 blocks remained active (67% of parameters wasted). The simple `resid_mix` mechanism (`h = mix0 * x + mix1 * x0` with unconstrained per-dim scalars) is strictly more expressive and naturally achieves the same quantization error reduction (Σ A\_i² = 2.20, 24% of standard residual) without any architectural overhead.


## Future Directions

**Lower-rank Q (r=128 or adaptive per-layer rank)**: r=128 showed 94-97% energy capture but crossed the quality threshold. An adaptive scheme — wider rank in deep layers (where attn\_scale peaks) and narrower in shallow layers — could push further.

**Better compilation for Pre-Rotation**: The content-dependent rotation achieved higher per-step quality but lost to compilation overhead. The core implementation is minimal:

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, n_rot_pairs=32):
        ...
        self.angle_proj = CastedLinear(dim, n_rot_pairs, bias=False)  # zero-init

    def forward(self, x):
        r = self.n_rot_pairs
        angles = self.angle_proj(x)          # [B, T, 32] content-dependent angles
        cos_a, sin_a = angles.cos(), angles.sin()
        x1, x2 = x[..., :r], x[..., r:2*r]
        x = torch.cat([
            x1 * cos_a + x2 * sin_a,         # rotated pair 1
            -x1 * sin_a + x2 * cos_a,         # rotated pair 2
            x[..., 2*r:]                       # unchanged dims
        ], dim=-1)
        return self.proj(torch.relu(self.fc(x)).square())
```

A custom Triton kernel fusing `angle_proj -> cos/sin -> rotation -> fc` into a single pass, or improvements in `torch.compile`'s handling of trigonometric operations within compiled graphs, would eliminate the 9-second overhead and make this technique viable. The method provides SwiGLU-like content-dependent feature mixing at 1% parameter cost with zero information loss (det(R)=1), making it particularly suited for parameter-constrained or inference-optimized settings.

## Training Configuration

```
NUM_LAYERS=11
WEIGHT_DECAY=0.038
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=524288
EVAL_STRIDE=64
MLP_MULT=3.0
MODEL_DIM=512
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
WARMDOWN_ITERS=3000
```
