# nGPT on the Hypersphere: Making Normalized Transformers Work at 16MB

**Not a record submission** — this is a research contribution documenting how to make nGPT (hypersphere-normalized transformers) viable under Parameter Golf constraints, including a novel fix for a torch.compile precision bug.

**val_bpb: 1.1502** (seed 1337, int6 sliding window stride=64, 8xH200 SXM)

## Summary

- **Made full nGPT trainable** at small scale by fixing three interacting bugs that caused PR #831 to dismiss it (BPB improved from 1.6915 → 1.2714)
- **Solved a torch.compile precision compounding bug** via opaque autograd function (`allow_in_graph`), enabling bf16 matmuls + fp32 normalizes with zero graph breaks — 25% faster than the fp32 workaround
- **Post-dequant renormalization** reduces nGPT's quantization gap from 0.35 → 0.008 BPB (44x reduction, 3 lines of code)
- **Re-enabled XSA + Partial RoPE** on nGPT for an additional -0.007 BPB (free improvement, zero overhead)
- **Systematic ablation** across 15+ configurations mapping the nGPT design space (layer count, MLP width, quantization, weight normalization, attention features, paper-faithfulness)

## Hardware

All results on **ORCD cluster** (MIT) using 8xH200 SXM (141GB HBM3e), CUDA 12.4, PyTorch 2.6.0. Training wallclock: 560 seconds (conservative budget leaving room for GPTQ). The bf16 compile fix (`L2NormalizeHP`) is compatible with PyTorch 2.6+ and RunPod 8xH100 SXM.

Step time on 8xH200: **119ms/step** (bf16 compile with opaque normalize). Estimated 8xH100: ~125ms/step.

## Best Result

Best configuration: Full nGPT, 12L 3x MLP, 512-dim, BigramHash 8192, XSA last 4, Partial RoPE 16, int6 + adaptive pruning.

| Seed | val_bpb (sliding) | artifact_bytes | steps | ms/step |
|------|-------------------|----------------|-------|---------|
| 1337 | **1.15018** | 15,889,025 | 4562 | ~121 |

3-seed validation on base config (without XSA/RoPE):

| Seed | val_bpb (sliding) | artifact_bytes | steps | ms/step |
|------|-------------------|----------------|-------|---------|
| 1337 | 1.15704 | 15,911,222 | 4646 | 120 |
| 42 | 1.15833 | 15,927,208 | ~4590 | 122 |
| 7 | 1.15937 | ~15,900,000 | ~4590 | 122 |
| **Mean** | **1.15825 ± 0.00117** | | | |

XSA + Partial RoPE add -0.007 BPB with zero speed cost (single-seed validated).

## Novel Contributions

### 1. Three Fixes That Make Full nGPT Trainable

PR #831 tested nGPT and got 1.6915 BPB — catastrophically bad. The diagnosis was "unit-norm weights incompatible with int6 quantization." We found three interacting bugs, each trivial to fix:

**Bug A: Zero-init + normalize = stuck at zero.** Standard zero-init on output projections (`proj._zero_init = True`) produces zero output. `F.normalize(0)` stays 0. The nGPT interpolation becomes `(1-α)*input + α*0 = (1-α)*input` — every block just decays the signal.

**Fix:** Small random init (std=0.01) for projection weights in nGPT mode.

**Bug B: Unit-norm hidden states → tiny logits.** With hidden states on the unit sphere, `F.linear(x, tok_emb.weight)` produces cosine similarities in [-1, 1]. Even with softcap=30, the model can't express confidence.

**Fix:** Learnable `logit_scale` parameter initialized to √dim ≈ 22.6.

**Bug C: Don't normalize token embeddings.** Normalizing embeddings destroys the magnitude information that the logit computation needs.

**Combined result:** 1.6915 → **1.2714 BPB** (only 0.015 behind standard model at equal params).

### 2. torch.compile Precision Compounding Bug (Novel Finding)

nGPT's forward pass executes ~86 L2 normalize calls. torch.compile's Inductor fuses bf16 operations, eliminating intermediate float32 casts in `F.normalize`. Each normalize introduces ~1e-3 error in bf16; across 86 sequential calls, errors compound to ~1e-2, causing catastrophic divergence from step 1.

**What we tried and why it failed:**

| Approach | Result | Root Cause |
|----------|--------|------------|
| Manual `x.float()` in normalize | Diverged (6.45) | Inductor fuses through float() casts |
| `emulate_precision_casts = True` | Diverged (6.43) | Not comprehensive for fused ops |
| `autocast(enabled=False)` locally | Diverged (1.70) | Inductor fuses across context managers |
| `@torch.compiler.disable` | Converges, 86+ graph breaks | Slower than eager mode |
| Custom Triton kernel | 17% slower | PyTorch ops already optimal at dim=512 |
| **Disable autocast globally (fp32)** | **Works, 126ms/step** | **Correct but slow** |

**The fix: `@torch._dynamo.allow_in_graph` on a custom `autograd.Function`.**

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

Key insight: `allow_in_graph` tells Dynamo to include the function as a single opaque node without tracing into it. Inductor never sees the internal float() casts, so it can't fuse them away. The function runs in eager Python with fp32 precision, while everything outside (matmuls, attention) runs in compiled bf16. Zero graph breaks.

| Mode | ms/step | Steps in 560s | Converges? |
|------|---------|---------------|------------|
| NO_COMPILE (eager) | 912ms | ~620 | Yes |
| Compile + fp32 forward | 126ms | ~4431 | Yes |
| **Compile + bf16 + L2NormalizeHP** | **119ms** | **~4706** | **Yes** |
| Standard model bf16 (reference) | 88ms | ~6400 | Yes |

This generalizes to any model with many sequential normalization steps under torch.compile. Related: [PyTorch Issue #168126](https://github.com/pytorch/pytorch/issues/168126).

### 3. Post-Dequantization Renormalization

nGPT's int6 quantization gap was 0.35 BPB (the reason PR #831 rejected nGPT). The fix is 3 lines:

```python
# After int6 dequantization:
if ngpt_quant == "renorm" and deq.ndim == 2 and "blocks." in name:
    deq = F.normalize(deq.float(), dim=-1).to(orig_dtype)
```

Int6 quantization adds magnitude noise to each weight row. Since nGPT weight rows should be unit-norm, renormalizing after dequant projects them back to the hypersphere.

| Condition | Quantization Gap |
|-----------|-----------------|
| Standard int6 dequant | +0.35 BPP |
| **Renorm dequant** | **+0.008 BPB** |
| Reduction | **44x** |

Critical requirement: **weight normalization during training** (`NGPT_WEIGHT_NORM=1`) is required for renorm dequant to work. Without it, weights aren't unit-norm and renormalization destroys learned scale information (quant gap becomes +1.6 BPB).

### 4. The Compression Paradox

Early nGPT results showed dramatically better compression (0.414 bytes/param at 1870 steps vs standard's 0.589). We discovered this is a mirage:

| Training Steps | Artifact (35.5M params) | bytes/param |
|---------------|------------------------|-------------|
| ~1870 (undertrained) | 14.7 MB | 0.414 |
| ~3640 (medium) | 17.8 MB | 0.501 |
| ~6300 (full) | 20.9 MB | 0.589 |
| Standard model (full training) | 15.9 MB (27M params) | **0.589** |

At full training length, nGPT compresses at exactly the same rate as standard models. The early advantage was from undertrained weights being close to orthogonal initialization (highly structured, compresses well). This has implications for any structured-weight approach: compression benefits measured at short training don't transfer.

### 5. Stochastic RYS (SRYS) on the Hypersphere

We developed Stochastic RYS: randomly repeat target layers during training with probability p, teaching layers to produce refinable representations for free depth at inference.

**On standard transformers:** -0.0005 BPB (the model learns near-identity repeats, cos_sim→0.999).
**On nGPT:** -0.006 BPB — **12x amplification.**

| Architecture | Baseline | + SRYS | Δ |
|-------------|----------|--------|---|
| Standard 512×11 | 1.2562 | 1.2557 | -0.0005 |
| nGPT-lite 512×11 | 1.3140 | 1.3081 | **-0.0059** |

The hypersphere constraint prevents the identity-collapse strategy (cos_sim can't reach 1.0 because unit-norm representations have geometric constraints). This forces the model to actually USE the second pass.

Note: Effect diminishes at higher param count (neutral at 35M+) and on full nGPT. Most effective on nGPT-lite at 27M params.

## Full Configuration Sweep

All runs: 8xH200 SXM, 560s wallclock, bf16 compile + L2NormalizeHP, seed 1337.

### Architecture sweep (base config: 12L 3x, no XSA/RoPE)

| Config | ms/step | Steps | Sliding BPB | Artifact | Fits? |
|--------|---------|-------|-------------|----------|-------|
| **12L 3x int6** | **120** | **4646** | **1.1570** | **15.9 MB** | **Yes** |
| 12L 3x int5 | 121 | ~4628 | 1.1647 | 13.1 MB | Yes |
| 12L 4x int5 | 130 | 4293 | 1.1599 | 14.9 MB | Yes |
| 12L 5x int5 | 137 | 4103 | 1.1720 | 15.0 MB | Yes |
| 11L 3x int6 | 110 | 5087 | 1.1736 | 14.3 MB | Yes |
| 11L 4x int5 | 118 | ~4735 | 1.1797 | 13.6 MB | Yes |
| 12L 3.5x int6 | 128 | ~4375 | 1.1754 | 16.1 MB | No |
| 13L 3x int5 | 128 | ~4375 | 1.1787 | 13.8 MB | Yes |
| 12L 3x no wn | 118 | ~4745 | 2.7842 | 16.6 MB | Broken |

### Feature stacking (on 12L 3x int6 base)

| Feature | Sliding BPB | Δ vs base | Cost |
|---------|-------------|-----------|------|
| Base (no XSA/RoPE) | 1.1570 | — | — |
| + XSA last 4 | 1.1532 | **-0.004** | Zero |
| + Partial RoPE 16 | 1.1525 | **-0.005** | Zero |
| **+ Both** | **1.1502** | **-0.007** | **Zero** |

### Paper-faithfulness experiments

| Change | Sliding BPB | Δ vs base | Lesson |
|--------|-------------|-----------|--------|
| Remove alpha `.abs()` | 1.3458 | +0.189 | Negative alpha needs >100K steps to learn |
| + s_z output scaling | Crashed | — | Extra params hurt at 5000 steps |
| + Constrained residual mix | (included above) | — | Unnecessary with subsequent normalize |

Key findings:
- **12L 3x is the sweet spot** — fewer layers lose quality, more layers lose speed
- **Int6 + adaptive pruning beats int5** by 0.008 BPB despite needing 7.8% pruning
- **Weight normalization is non-negotiable** — removing it causes catastrophic quant failure (+1.6 BPB)
- **XSA + Partial RoPE are free wins** — -0.007 BPB combined, zero overhead
- **Paper design choices hurt at short training** — `.abs()`, s_z, constrained mix all regress
- **More params don't help** — the step-time cost outweighs the per-step quality gain

## Architecture

```
Token Embeddings → BigramHash (8192) → 12× nGPT Blocks → Tied Head
                                           ↓
                                    normalize(input)
                                    attention(h_norm)
                                    normalize(attn_out)
                                    slerp(h_norm, attn_norm, α_a)  ← hypersphere interpolation
                                    mlp(x_out)
                                    normalize(mlp_out)
                                    slerp(x_out, mlp_norm, α_m)    ← hypersphere interpolation
```

| Component | Setting |
|-----------|---------|
| Layers | 12 |
| Model dim | 512 |
| Attention | GQA 8 heads / 4 KV heads |
| MLP | 3x expansion, LeakyReLU(0.5)² |
| BigramHash | 8192 vocab |
| U-Net skip connections | Yes |
| XSA | Last 4 layers |
| Partial RoPE | 16 of 64 head dims |
| Normalization | Full nGPT (L2 normalize both sides) |
| Weight normalization | Yes (forward-pass row-wise L2 norm) |
| Parameters | 30M |

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Muon (matrix_lr=0.025, WD=0.04, momentum=0.99) + AdamW (embed_lr=0.035, scalar_lr=0.025) |
| Batch | 786K tokens |
| Sequence length | 2048 |
| Warmup | 20 steps |
| Warmdown | 3500 iters |
| SWA | Last ~10% of warmdown |
| Quantization | Int6 QAT with STE + Full GPTQ + renorm dequant |
| Compression | zstd level 22 |
| Pruning | Adaptive Hessian-weighted (~7.8% to fit 16MB) |
| Eval | Sliding window, stride=64 |

## Interpretability (Layer Analysis)

We performed logit lens, layer knockout, and CKA similarity analysis on our 11L standard model to understand the layer structure:

| Layer | BPB (logit lens) | Knockout Δ | Function |
|-------|-----------------|------------|----------|
| 0 | 5.46 (-4.88) | +2.31 | Basic token patterns (most critical) |
| 1-2 | 3.82 | +0.58-0.65 | Pattern matching (encoder) |
| 3-5 | 5.29 (+1.47) | +0.15-0.29 | U-Net transition (BPP **increases**) |
| 6-7 | 2.57 (-2.72) | +0.18-0.19 | Critical "thinking" layers |
| 8-10 | 1.19 (-1.38) | +0.14-0.16 | Prediction refinement |

Three functional circuits: Encoder (L0-2, CKA 0.83), Transition (L3-6), Prediction (L7-10, CKA 0.88). The prediction circuit is the tightest — layers 7-8-9 work as a single unit.

## What We Tried That Didn't Work

| Idea | Result | Lesson |
|------|--------|--------|
| RYS (layer repetition) at eval | +0.021 to +0.638 BPB damage | Layers too specialized at 27M params |
| Stochastic RYS on standard model | -0.0005 BPB (negligible) | Model learns to make repeats identity |
| Riemannian Muon (tangent-plane projection) | 0.19 BPB behind at 2000 steps | Convergence speed dominates at short training |
| Triton fused normalize kernel | 17% slower than PyTorch ops | Kernel launch overhead > compute savings at dim=512 |
| No weight normalization | +1.6 BPB quant failure | Renorm dequant requires unit-norm weights |
| 13L / wider MLP (more params) | All worse than 12L 3x | Step-time cost > per-step quality gain |
| Paper-faithful alpha (no `.abs()`) | +0.189 BPB | Negative alpha needs >100K steps to learn |
| Paper-faithful s_z output scaling | Crashed (SDPA kernel error) | Extra params + overhead at 5000 steps |
| TTT (any LR, with/without renorm) | NaN on all configs | GPTQ + renorm dequant weights are too fragile for gradient updates |

### TTT Incompatibility (Important Negative Result)

Test-time training produces NaN on nGPT models regardless of:
- Learning rate (tested 0.00005 to 0.002)
- With or without weight renormalization after TTT steps
- Freezing 10/12 blocks (only last 2 unfrozen)

The root cause: renorm dequantization during the forward pass creates a numerically fragile computational graph. Gradients flowing through the dequant→renormalize→forward path become unstable. This is specific to the combination of Full GPTQ + renorm dequantization — standard GPTQ (without renorm) tolerates TTT at very low LRs.

## Discussion

**Gap to SOTA:** Our best nGPT result (1.1502) is 0.031 behind SOTA (1.1194). Two factors:
1. **Step time:** nGPT at 121ms/step gets ~4600 steps vs the standard model's ~6800 at 88ms/step. The 33ms overhead from ~86 opaque normalize calls is a fundamental architectural cost.
2. **No TTT:** GPTQ + renorm dequant weights are incompatible with gradient-based adaptation. SOTA uses TTT for ~-0.002 BPB.

**Research value:** The contributions here aren't about BPB — they're about understanding what happens when you constrain representations to the hypersphere under extreme compression:

1. **nGPT was dismissed too early.** Three trivial initialization fixes close the gap from 0.43 to 0.015 BPB.
2. **Post-dequant renormalization is a general technique.** Any unit-norm weight model benefits from re-projecting after quantization.
3. **torch.compile has a precision compounding bug.** The `allow_in_graph` fix applies to any model with many sequential normalizations.
4. **Compression advantages from structured weights vanish at full training.** This is a cautionary result for the weight-sharing community.
5. **RYS amplification on the hypersphere** is a genuinely novel interaction — the geometry prevents identity collapse, enabling 12x stronger layer repetition effects.
6. **Paper design choices don't transfer to short training.** Signed alpha, s_z scaling, and other nGPT paper features that improve long-training performance actually hurt at 5000 steps — the optimizer doesn't have time to exploit the extra degrees of freedom.

## References

- [nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/abs/2410.01131) (Loshchilov et al., ICLR 2025)
- [Parameter Golf PR #831: Why Novel Architectures Fail at 16MB](https://github.com/openai/parameter-golf/pull/831)
- [Parameter Golf PR #579: The Frugendorff](https://github.com/openai/parameter-golf/pull/579) — weight sharing research
- [dnhkng RYS blog](https://dnhkng.github.io/posts/rys/) — original RYS technique on Qwen2-72B
- [PyTorch Issue #168126](https://github.com/pytorch/pytorch/issues/168126) — torch.compile precision divergence
