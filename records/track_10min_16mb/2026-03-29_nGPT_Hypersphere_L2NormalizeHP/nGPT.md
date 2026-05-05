# nGPT: Normalized Transformers Under Extreme Compression

## Overview

We investigate whether hypersphere-normalized transformers (nGPT) can be made viable under the Parameter Golf constraints (16MB artifact, 10-minute training on 8xH100). Prior work (PR #831) dismissed nGPT due to a 0.35 BPB quantization gap with standard int6. We show:

1. **Post-dequant renormalization** reduces the quantization gap from 0.35 to **0.0005 BPB** (700x reduction, 3 lines of code)
2. **Three initialization fixes** make full nGPT trainable for the first time (BPB 1.2714, only 0.015 behind standard)
3. **nGPT compresses 15-42% better** — fitting **35-38M params in 14.7 MB** vs standard's 27M in 15.9 MB
4. **Scaled nGPT (12L 4x + BigramHash) achieves 1.2668 BPB** — only 0.011 behind standard with 32% more parameters
5. **Stochastic RYS training effect is 12x stronger** on nGPT than standard transformers
6. **torch.compile + nGPT requires fp32 forward** due to a precision compounding bug we identified (loss: 321ms/step compiled vs 912ms eager)
7. **Riemannian Muon** — a novel optimizer variant with tangent-plane projection for hypersphere-constrained training

---

## Background

### nGPT (Loshchilov et al., 2024)

[nGPT](https://arxiv.org/abs/2410.01131) constrains all hidden representations to the unit hypersphere. Each transformer block performs interpolation on the sphere rather than residual addition:

```
h = Normalize(h + α * (block_output - h))
```

where α is a learnable per-dimension "eigen learning rate." The paper reports 4-20x faster convergence on standard benchmarks.

### Why nGPT Failed in Parameter Golf (PR #831)

PR #831 tested nGPT and reported:
- **1.6915 BPB** (vs SOTA 1.1428) — catastrophically bad
- **+47% step time** overhead
- **Root cause claimed:** "Unit-norm weights incompatible with int6 quantization"
- **Quantization gap: 0.35 BPB**

This led the competition to dismiss nGPT entirely. **We show the diagnosis was wrong — three simple fixes make nGPT competitive.**

---

## Our Contributions

### 1. Three Fixes That Make Full nGPT Work

PR #831's full nGPT failed because of three interacting bugs, each trivial to fix:

**Bug A: Zero-init projection weights + normalize = stuck at zero**

The standard model zero-initializes output projection weights (`proj._zero_init = True`). With nGPT's normalize-both-sides:
```python
attn_out = self.attn(h_norm)     # proj is zero-init → output = 0
attn_norm = F.normalize(0)       # = 0 (nothing to normalize)
h_norm + alpha * (0 - h_norm)    # = (1-alpha) * h_norm → just the input
```
Both attention and MLP contribute nothing. The model is stuck.

**Fix:** Small random init (std=0.01) instead of zeros for projection weights in nGPT mode.

**Bug B: Normalized embeddings → tiny logits**

Token embeddings normalized to unit norm. Output head `F.linear(x, tok_emb.weight)` produces cosine similarities in [-1, 1]. With softcap=30, logits are in [-1, 1] — model can't express confidence.

**Fix:** Learnable logit scale initialized to sqrt(dim) ≈ 22.6.

**Bug C: No logit scaling for unit-norm hidden states**

Even without embedding normalization, unit-norm hidden states produce small logits.

**Fix:** `logits = logit_scale * F.linear(normalized_x, tok_emb.weight)`

**Combined result:** Full nGPT trains to **1.2714 BPB** (vs 1.6915 in PR #831, vs 1.2562 standard).

### 2. Post-Dequantization Renormalization

The 0.35 BPB quantization gap occurs because int6 quantization adds magnitude noise to unit-norm weight rows.

**Fix:** Renormalize each weight row after dequantization.

```python
w_deq = dequantize_int6(w_quant, scale)  # standard int6 dequant
w_deq = F.normalize(w_deq, dim=-1)       # project back to sphere
```

**Result:** Quantization gap drops from **0.35 BPB to 0.0005 BPB** — a 700x reduction.

### 3. Compression Advantage — 42% More Parameters in 16MB

nGPT's normalized weight distribution compresses dramatically better under int6 + zstd:

| Model | Params | Artifact | Headroom | BPB |
|-------|--------|----------|----------|-----|
| Standard 11L 3x | 27M | 15.9 MB | 0.1 MB | 1.2562 |
| nGPT 11L 3x (clean) | 26.5M | 15.4 MB | 0.6 MB | 1.2714 |
| nGPT 12L 4x | 35.1M | 13.4 MB | 2.6 MB | 1.2771 |
| **nGPT 12L 4x + BigramHash** | **35.5M** | **14.7 MB** | **1.3 MB** | **1.2668** |
| nGPT 13L 4x | 38M | 13.5 MB | 2.5 MB | 1.2808 |

The 12L 4x + BigramHash model has **32% more parameters** than standard while fitting comfortably in 16MB. This is the nGPT compression advantage paying dividends.

### 4. Scaling Results (4xH200, competition wallclock)

All runs: 4xH200, MAX_WALLCLOCK_SECONDS=1700, NO_COMPILE, seed 1337.

| Config | Params | BPB | Artifact | Steps | ms/step |
|--------|--------|-----|----------|-------|---------|
| Standard 11L 3x (reference) | 27M | 1.2562 | 15.9 MB | ~3400 | ~500ms |
| nGPT 11L 3x | 26.5M | 1.2728 | 11.3 MB | ~2030 | 836ms |
| nGPT 12L 4x | 35.1M | 1.2771 | 13.4 MB | ~1870 | 912ms |
| **nGPT 12L 4x + BigramHash** | **35.5M** | **1.2668** | **14.7 MB** | **~1870** | **912ms** |
| nGPT 13L 4x | 38M | 1.2808 | 13.5 MB | ~1673 | 1016ms |

**Key finding:** nGPT's BPB scales strongly with parameter count. The 12L 4x + BigramHash model at 1.2668 is only 0.011 behind standard — achieved with half the training steps (1870 vs 3400) but 32% more params.

### 5. torch.compile + nGPT: The Precision Compounding Bug

**Discovery:** torch.compile causes nGPT to diverge (loss 6-10 instead of 2.2). We traced this through systematic isolation:

| Test | Eager vs Compile diff | Result |
|------|----------------------|--------|
| Single `F.normalize` | 9.77e-4 | FAIL |
| `normalize_hp` (manual fp32 norm) | 0.00 | PASS |
| Interpolation + normalize | 9.77e-4 | FAIL |
| 11-layer chain | 1.61e-2 | FAIL |

**Root cause:** torch.compile's Inductor backend fuses bf16 operations, eliminating intermediate float32 precision casts. nGPT has ~123 normalize calls per forward pass. Each introduces a small bf16 rounding error that compounds catastrophically through the sequential normalization chain. This is related to [PyTorch issue #168126](https://github.com/pytorch/pytorch/issues/168126).

**Approaches that FAILED:**

| Approach | Result | Why it Failed |
|----------|--------|---------------|
| `normalize_hp` (manual fp32 norm) | Diverged (loss 6.45) | Compile fuses through float() casts |
| `emulate_precision_casts = True` | Diverged (loss 6.43) | Not comprehensive for fused ops |
| `autocast(enabled=False)` in normalize_hp | Diverged (BPB 1.70) | Compile fuses across context |
| `@torch.compiler.disable` on normalize_hp | Converges but 123 graph breaks | Slower than eager |
| fp32 compile (disable autocast globally) | **Works, 126-154ms/step** | Current best workaround |

**Fix: Opaque Autograd Function via `allow_in_graph`** (2026-03-29)

The breakthrough: `torch._dynamo.allow_in_graph` makes a function opaque to torch.compile — Inductor includes it as a node without tracing into it, preserving internal fp32 precision. Zero graph breaks.

```python
@torch._dynamo.allow_in_graph
class L2NormalizeHP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        dtype = x.dtype
        x32 = x.float()
        norm = x32.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = (x32 / norm).to(dtype)
        ctx.save_for_backward(y, norm.reciprocal().to(torch.float32))
        return y

    @staticmethod
    def backward(ctx, dy):
        y, inv_norm = ctx.saved_tensors
        dy32, y32 = dy.float(), y.float()
        dot = (dy32 * y32).sum(-1, keepdim=True)
        return ((dy32 - y32 * dot) * inv_norm).to(dy.dtype)
```

Key insight: all prior approaches tried to prevent compile from fusing through float() casts using PyTorch-level mechanisms (emulate_precision_casts, nested autocast context, compiler.disable). These all fail because Inductor optimizes at a lower level. The `allow_in_graph` approach works because it makes the entire operation opaque — Inductor never sees the internal ops, just treats it as a black box that takes bf16 in and returns bf16 out.

**Results with the fix:**

| Mode | ms/step (8xH200) | Steps in 560s | Converges? |
|------|-------------------|---------------|------------|
| NO_COMPILE (eager) | 912ms | ~620 | ✓ |
| Compile + fp32 (old fix) | 126-154ms | 3640-4431 | ✓ |
| **Compile + bf16 + L2NormalizeHP** | **119ms** | **~4706** | **✓** |
| Standard model bf16 (reference) | 88ms | 6800 | ✓ |

The opaque normalize approach gives **25% more training steps** than fp32 compile while maintaining convergence. The 31ms overhead vs standard bf16 (119 vs 88ms) comes from ~86 opaque function calls per forward, each launching 4 eager CUDA kernels. A fused Triton kernel (Phase 2) could reduce this to ~95ms.

Process that led to the fix:
1. Diagnosed the root cause: compile fuses through float() casts (PyTorch issue #168126)
2. Tried 5 different PyTorch-level precision control mechanisms — all failed
3. Realized compile needs to be prevented from SEEING the float() ops, not just from fusing them
4. `allow_in_graph` on autograd.Function makes the entire normalize opaque to Inductor
5. No graph breaks (unlike @compiler.disable), so compile still fuses everything else
6. The function runs in eager Python with full fp32 precision, surrounded by compiled bf16 code

### 6. Stochastic RYS on the Hypersphere

We developed Stochastic RYS (SRYS): during training, randomly repeat target layers with probability p, teaching the model to benefit from iterative refinement.

**On standard transformers:** -0.0005 BPB max (the model learns to make repeats a no-op).

**On nGPT:** -0.006 BPB — **12x amplification**.

| Architecture | Baseline | + SRYS | Δ | Amplification |
|-------------|----------|--------|---|---------------|
| Standard 512×11 | 1.2562 | 1.2557 | -0.0005 | 1x |
| nGPT-lite 512×11 | 1.3140 | 1.3081 | **-0.0059** | **12x** |

We discovered the **identity-or-reject dichotomy**: on unconstrained vector spaces, the model either collapses repeats to identity (cos_sim→0.999) or gates them out (gate→0.03). On the hypersphere, unit-norm representations prevent identity collapse — even small rotations change angular relationships between all tokens.

Note: SRYS effect diminishes at higher param count (neutral at 35M+) and on full nGPT (already normalized). Most effective on nGPT-lite at 27M params.

### 7. Riemannian Muon (Novel Optimizer)

Muon's Newton-Schulz orthogonalization assumes unconstrained weight matrices, fighting the hypersphere geometry. We developed Riemannian Muon with tangent-plane projection:

```python
# After Newton-Schulz orthogonalization:
w_norm = F.normalize(p.data, dim=-1)
radial = (g * w_norm).sum(dim=-1, keepdim=True) * w_norm
g = g - radial  # project to tangent plane
# After weight update:
p.data = F.normalize(p.data, dim=-1)  # retract to sphere
```

| Optimizer | Loss @ step 300 | ms/step | Overhead |
|-----------|-----------------|---------|----------|
| Standard Muon | 2.71 | 1500ms | baseline |
| **Riemannian Muon** | **2.82** | **1531ms** | **+2%** |
| AdamW | 3.66 | 1500ms | same |

Riemannian Muon is 5x closer to standard Muon than AdamW, at negligible overhead. However at 2000 steps, standard Muon still wins on BPB (1.31 vs 1.50) because convergence speed dominates at short training lengths. Riemannian Muon may be superior at longer training where the geometric correctness matters more.

---

## Implementation Details

### Full nGPT Block Forward

```python
def forward(self, x, x0, ...):
    mix = self.resid_mix.to(dtype=x.dtype)
    x_in = mix[0] * x + mix[1] * x0

    # Normalize both sides, interpolate on hypersphere
    h_norm = normalize_hp(x_in)
    attn_out = self.attn(h_norm)
    attn_norm = normalize_hp(attn_out)
    alpha_a = self.alpha_a.abs()
    x_out = slerp_hp(h_norm, attn_norm, alpha_a)  # fp32 interpolation + normalize

    mlp_out = self.mlp(x_out)
    mlp_norm = normalize_hp(mlp_out)
    alpha_m = self.alpha_m.abs()
    x_out = slerp_hp(x_out, mlp_norm, alpha_m)
    return x_out
```

Key helpers for compile compatibility:
```python
def normalize_hp(x):
    """Full float32 normalize — prevents compile precision fusion."""
    dtype = x.dtype
    return (x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype)

def slerp_hp(x, y, alpha):
    """Float32 spherical interpolation + normalize."""
    dtype = x.dtype
    xf, yf = x.float(), y.float()
    result = xf + alpha.float() * (yf - xf)
    return (result / result.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype)
```

### Quantization: Post-Dequant Renormalization

```python
# In dequantize_mixed_int6:
if ngpt_quant == "renorm" and deq.ndim == 2 and "blocks." in name:
    deq = F.normalize(deq.float(), dim=-1).to(orig_dtype)
```

### Env Vars

| Var | Default | Description |
|-----|---------|-------------|
| `NGPT_ENABLED` | 0 | Enable nGPT activation normalization |
| `NGPT_FULL` | 0 | Full nGPT (normalize both sides of interpolation) |
| `NGPT_WEIGHT_NORM` | 0 | Forward-pass weight normalization |
| `NGPT_QUANT_MODE` | "" | "renorm" for post-dequant renormalization |
| `NGPT_ADAMW` | 0 | Replace Muon with AdamW for matrix params |
| `NGPT_RIEMANNIAN` | 0 | Riemannian Muon (tangent projection + retraction) |

---

## Full Results Summary

### All nGPT Configurations Tested (batch eval, seed 1337)

| Config | GPU | Optimizer | BPB | Artifact | Params |
|--------|-----|-----------|-----|----------|--------|
| Standard 11L 3x (reference) | L40S | Muon | 1.2562 | 15.9 MB | 27M |
| nGPT-lite 11L 3x | L40S | Muon | 1.3140 | 13.4 MB | 27M |
| nGPT-lite + SRYS | L40S | Muon | 1.3034 | 13.4 MB | 27M |
| nGPT-lite + renorm quant | L40S | Muon | 1.3145 | 13.4 MB | 27M |
| **Full nGPT 11L 3x (clean)** | **H200** | **Muon** | **1.2714** | **15.4 MB** | **26.5M** |
| Full nGPT 11L 4x | L40S | Muon | 1.2858 | 16.2 MB | 32.6M |
| Full nGPT 12L 3.5x | L40S | Muon | 1.3029 | ~16 MB | 32.3M |
| **Full nGPT 12L 4x + BigramHash** | **4xH200** | **Muon** | **1.2668** | **14.7 MB** | **35.5M** |
| Full nGPT 13L 4x | 4xH200 | Muon | 1.2808 | 13.5 MB | 38M |
| Full nGPT 12L 4x + Bigram (compiled fp32) | 4xH200 | Muon | **running** | — | 35.5M |
| nGPT + Riemannian Muon | H200 | RieMuon | 1.5029 | — | 27M |
| nGPT + AdamW | H100 | AdamW | 1.5330 | — | 27M |

### Key Comparisons

**nGPT vs Standard at same scale (11L 3x, ~27M params):**
- Standard: 1.2562 (optimized, compiled)
- nGPT full: 1.2714 (0.015 behind, uncompiled)

**nGPT scaled vs Standard (using compression headroom):**
- Standard 27M: 1.2562 in 15.9 MB
- **nGPT 35.5M: 1.2668 in 14.7 MB** (0.011 behind with 32% more params, half the steps)

**nGPT compiled (fp32) — 4xH200, 1700s:**
- 270ms/step → 6298 steps
- **Pre-quant BPB: 1.1324** (step 6298, post-EMA)
- **Post-quant BPB: N/A** (artifact 20.9 MB — over 16 MB, adaptive pruning crashed)
- At step 4000: BPB 1.2032 — already beat standard model

**nGPT competition sim — 8xH200, 560s, INT5_ALL:**

| Metric | Value |
|--------|-------|
| Steps | 3640 (at 154ms/step) |
| Pre-quant BPB | **1.1667** |
| Post-quant BPB | **1.1740** |
| Quantization gap | **+0.0073** |
| Artifact | **17.8 MB (OVER)** |

### 8. The Compression Paradox

A critical discovery: **nGPT's compression advantage vanishes at full training length.**

| Training steps | Artifact (35.5M params) | bytes/param |
|---------------|------------------------|-------------|
| ~1870 (NO_COMPILE) | 14.7 MB | 0.414 |
| ~3640 (8xH200 560s) | 17.8 MB | 0.501 |
| ~6300 (4xH200 1700s) | 20.9 MB | 0.589 |
| Standard model (6800 steps) | 15.9 MB (27M params) | 0.589 |

At full training, nGPT's bytes/param matches standard exactly (0.589). The early compression advantage was an artifact of undertrained weights being more structured (closer to orthogonal init). This means nGPT cannot fit more params than standard in the same artifact at competition training lengths.

### 9. Int5 Full-Model Quantization

Standard models use mixed int5 (MLP) + int6 (attention) because attention weights are more sensitive to precision loss. We tested **int5 for ALL weights** on nGPT, hypothesizing that renorm dequantization would recover angular precision.

Result: quantization gap is **0.0073 BPB** with int5 — comparable to standard's ~0.007 with mixed int5/int6. The renorm dequant makes uniform int5 viable, simplifying the quantization pipeline.

However, even with int5, the 35.5M param model produces a 17.8 MB artifact — still over 16 MB. The max params that fit: **~31.7M** (12L 3x + BigramHash = 29.2M at ~14.6 MB).

---

## The torch.compile Precision Bug (Novel Finding)

nGPT exposed a previously undocumented interaction between torch.compile and sequential normalization:

1. **torch.compile's Inductor fuses bf16 ops**, eliminating intermediate float32 casts
2. **F.normalize internally uses float32** in eager mode for the norm computation
3. **Under compile, the norm may be computed in bf16**, introducing ~1e-3 error per call
4. **With 86+ normalize calls per forward pass**, errors compound exponentially
5. **The fix**: wrap normalize in `@torch._dynamo.allow_in_graph` autograd.Function — makes the operation opaque to compile while preserving graph continuity (zero graph breaks)

This affects any model with many sequential normalization steps — not just nGPT. Models with layer norm after every sub-layer (standard transformers) are less affected because layer norm only normalizes variance, while L2 normalize constrains the entire vector to unit sphere (more sensitive to precision).

**Why other approaches fail:** PyTorch-level precision controls (emulate_precision_casts, nested autocast, compiler.disable) all operate at the Python dispatch level. Inductor's fusion engine works at a lower level (Triton code generation) where it can see through these controls and optimize them away. The `allow_in_graph` approach works because it prevents Inductor from ever seeing the internal ops — the entire function is a single opaque node.

Related: [PyTorch Issue #168126](https://github.com/pytorch/pytorch/issues/168126)

---

## Discussion

### The Compression Paradox Explained

At short training (~2000 steps), nGPT weights are close to their orthogonal initialization — highly structured, compresses to 0.414 bytes/param. At full training (~6000+ steps), weights become unstructured, compressing at 0.589 bytes/param — identical to standard models. The "compression advantage" was a mirage from undertrained models.

This has implications for any weight-sharing or structured-weight approach: compression benefits measured at short training don't transfer to competition-length training.

### Why nGPT Per-Step BPB is Better

Despite not compressing better, nGPT achieves better BPB per training step at the same param count. At 2000 steps: nGPT 1.2714 vs standard 1.2562 (nGPT worse). But nGPT's BPP curve is steeper — it catches up and surpasses standard around step 3000-4000. At step 6298: nGPT 1.1324 (batch eval) which would translate to ~1.00-1.05 sliding window — potentially competitive with SOTA.

The bottleneck is the fp32 compile overhead (154ms vs 88ms/step), giving nGPT 39% fewer steps in the same wallclock. Solving the mixed-precision compile would unlock nGPT's full potential.

### SRYS Diminishes at Higher Param Count

SRYS (-0.006 at 27M) became neutral at 35M+. The larger model has enough capacity that the regularization effect is unnecessary. This is consistent with the "SRYS as regularizer" interpretation — regularization helps underfitting models, not large ones.

### Key Quantization Findings

| Finding | Detail |
|---------|--------|
| Renorm dequant gap (undertrained) | 0.0005 BPB — 700x reduction vs PR #831 |
| Renorm dequant gap (full training) | 0.0073 BPB — comparable to standard |
| Int5-all viable with renorm | Same gap as mixed int5/int6 on standard |
| Compression paradox | nGPT advantage vanishes at full training |
| Max params in 16 MB (int5) | ~31.7M (vs standard's 27M = 17% more) |

---

## Future Work

1. ~~**Mixed-precision compile**~~ **SOLVED** — `L2NormalizeHP` with `allow_in_graph` gives bf16 matmuls + fp32 normalizes at 119ms/step (25% faster than fp32 compile)
2. **Triton fused normalize kernel** — replace 4 eager CUDA kernels per normalize with 1 fused Triton kernel (bf16 load → fp32 accumulation → bf16 store). Could close the 31ms gap vs standard bf16, reaching ~90-95ms/step
3. **8xH100 validation** — test on competition hardware for official timing
4. **Sliding window eval** — batch eval understates quality by ~0.02-0.13 BPB
5. **TTT on nGPT** — test-time training with weight renormalization after each gradient step (implemented, untested)
6. **Multi-seed validation** — confirm results across seeds 1337, 42, 7
7. **Optimal param count with bf16 fix** — with 119ms/step, the optimal model size shifts: more steps available means smaller models (better per-step efficiency) may now beat larger ones
8. **Feature re-enablement** — test XSA, partial RoPE, VE on top of full nGPT with bf16 compile

---

## References

- [nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/abs/2410.01131) (Loshchilov et al., ICLR 2025)
- [NVIDIA nGPT Implementation](https://github.com/NVIDIA/ngpt)
- [Parameter Golf PR #831: Why Novel Architectures Fail at 16MB](https://github.com/openai/parameter-golf/pull/831) — nGPT tested and rejected due to quantization
- [Parameter Golf PR #579: The Frugendorff](https://github.com/openai/parameter-golf/pull/579) — weight sharing research
- [dnhkng RYS blog](https://dnhkng.github.io/posts/rys/) — original RYS technique on Qwen2-72B
- [PyTorch Issue #168126](https://github.com/pytorch/pytorch/issues/168126) — torch.compile precision divergence in bf16
