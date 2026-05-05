# nGPT on the Hypersphere: Making Normalized Transformers Work at 16MB

This is a research contribution, not a record submission. I wanted to see if nGPT (Loshchilov et al., ICLR 2025) could be made competitive under Parameter Golf constraints after PR #831 dismissed it. It can — and the investigation uncovered some interesting findings about torch.compile, quantization, and what happens when you try to squeeze a hypersphere-constrained architecture into 16MB.

**val_bpb: 1.1502** (seed 1337, int6 sliding window stride=64, 8xH200 SXM, 560s)

## What I found

1. **PR #831's nGPT failure (1.69 BPB) was caused by three init bugs, not architectural incompatibility.** Fixing them takes nGPT to 1.27 BPB — only 0.015 behind the standard model at equal params.
2. **torch.compile has a precision compounding bug with sequential L2 normalization.** nGPT's ~86 normalize calls per forward cause bf16 errors to compound catastrophically. I found a fix using `allow_in_graph` that preserves fp32 precision with zero graph breaks. This applies to any model with many sequential normalizations. (Related: pytorch/pytorch#168126)
3. **Renormalizing weight rows after int6 dequantization cuts nGPT's quantization gap from +0.35 to +0.008 BPB.** Three lines of code, 44x improvement. Weight normalization during training is required — without it the quant gap balloons to +1.6 BPB.
4. **nGPT's compression advantage is a mirage.** At short training, normalized weights compress to 0.41 bytes/param vs standard's 0.59. At full training length, both converge to 0.59. The early advantage comes from undertrained weights sitting near their orthogonal initialization.
5. **The nGPT paper's design choices (signed alpha, s_z scaling) hurt at 5000 steps.** These features need long training (100K+ steps) to be useful. At our budget, constraining alpha to be positive and skipping s_z scaling actually works better.

## Hardware

All experiments on MIT ORCD cluster, 8xH200 SXM (141GB HBM3e), CUDA 12.4, PyTorch 2.6. Training wallclock capped at 560s. The `L2NormalizeHP` fix works on PyTorch 2.6+ including RunPod's 2.9.

## Results

Best config: 12L, 512d, 3x MLP, GQA 8/4, BigramHash 8192, XSA on last 4 layers, Partial RoPE (16 of 64 dims), int6 QAT + Full GPTQ + renorm dequant + adaptive pruning.

| Seed | val_bpb (sliding) | artifact | steps | ms/step |
|------|-------------------|----------|-------|---------|
| 1337 | **1.1502** | 15.9 MB | 4562 | 121 |

3-seed validation on the base config (no XSA/RoPE):

| Seed | val_bpb (sliding) | artifact |
|------|-------------------|----------|
| 1337 | 1.1570 | 15.9 MB |
| 42 | 1.1583 | 15.9 MB |
| 7 | 1.1594 | 15.9 MB |
| **Mean ± std** | **1.1582 ± 0.0012** | |

Adding XSA + Partial RoPE gives another -0.007 BPB at zero cost.

## The three init fixes (PR #831 revisited)

PR #831 tested nGPT and reported 1.69 BPB — concluding nGPT is incompatible with int6 quantization. The actual problems were simpler:

**A. Zero-init projections + normalize = dead blocks.** Output projections are zero-initialized. `F.normalize(0) = 0`. The interpolation `h + α*(0 - h)` just decays the input. Every block does nothing.
→ Fix: small random init (std=0.01) for projection weights.

**B. Unit-norm hidden states produce tiny logits.** `F.linear(x, emb.weight)` gives cosine similarities in [-1, 1]. The model can't express confidence.
→ Fix: learnable `logit_scale` initialized to √dim ≈ 22.6.

**C. Don't normalize token embeddings.** The logit head needs embedding magnitude to distinguish tokens.

Together: 1.69 → **1.27 BPB**.

## The torch.compile precision bug

This was the hardest problem and the most generalizable finding.

nGPT runs ~86 `F.normalize` calls per forward pass. Each one internally upcasts to float32 for the norm computation. Under `torch.compile`, Inductor fuses these ops and eliminates the intermediate float32 casts. Each normalize picks up ~1e-3 bf16 rounding error. Across 86 sequential calls, errors compound to ~1e-2 — the model diverges from step 1.

I tried five different approaches before finding one that works:

| Approach | Result | Why it failed |
|----------|--------|---------------|
| Manual `x.float()` in the normalize function | Diverged | Inductor fuses through the cast |
| `emulate_precision_casts = True` | Diverged | Doesn't cover fused matmul accumulation |
| Nested `autocast(enabled=False)` | Diverged | Inductor fuses across context boundaries |
| `@torch.compiler.disable` on normalize | Converged but 86 graph breaks | Slower than not compiling at all |
| Fused Triton kernel (bf16 in, fp32 accum, bf16 out) | Converged | 17% slower than PyTorch ops at dim=512 |

**What worked:** wrapping normalize in a `torch.autograd.Function` decorated with `@torch._dynamo.allow_in_graph`. This tells Dynamo to treat the function as a single opaque node — Inductor never sees the internal ops, so it can't fuse away the float32 precision. Everything outside (matmuls, attention, elementwise) still runs in compiled bf16. Zero graph breaks.

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

| Mode | ms/step | Steps in 560s |
|------|---------|---------------|
| Eager (no compile) | 912 | ~620 |
| Compile, fp32 forward (workaround) | 126 | ~4400 |
| **Compile, bf16 + L2NormalizeHP** | **119** | **~4700** |
| Standard model, bf16 compile | 88 | ~6400 |

The remaining 31ms gap vs the standard model comes from 86 opaque Python function calls per forward — each exits the compiled graph, runs 4 eager CUDA kernels, and re-enters. This is the fundamental cost of the approach.

## Post-dequant renormalization

The original 0.35 BPB quantization gap that sank PR #831 comes from int6 adding magnitude noise to unit-norm weight rows. Renormalizing each row after dequantization projects the weights back onto the hypersphere:

```python
if ngpt_quant == "renorm" and deq.ndim == 2 and "blocks." in name:
    deq = F.normalize(deq.float(), dim=-1).to(orig_dtype)
```

Quant gap drops from +0.35 to +0.008 BPB. But this only works if weights are actually unit-norm — which requires `NGPT_WEIGHT_NORM=1` during training. Without it, renormalization destroys learned scale information and the gap shoots to +1.6 BPB.

## Configuration sweep

I ran 15+ configurations to map the design space. All on 8xH200, 560s, bf16 compile.

### Model size (base config, no XSA/RoPE)

| Config | ms/step | Steps | BPB | Artifact | Fits? |
|--------|---------|-------|-----|----------|-------|
| **12L 3x int6** | **120** | **4646** | **1.1570** | **15.9 MB** | **Yes** |
| 12L 4x int5 | 130 | 4293 | 1.1599 | 14.9 MB | Yes |
| 12L 5x int5 | 137 | 4103 | 1.1720 | 15.0 MB | Yes |
| 11L 3x int6 | 110 | 5087 | 1.1736 | 14.3 MB | Yes |
| 11L 4x int5 | 118 | ~4735 | 1.1797 | 13.6 MB | Yes |
| 12L 3.5x int6 | 128 | ~4375 | 1.1754 | 16.1 MB | No |
| 12L 3x, no weight norm | 118 | ~4745 | 2.7842 | — | Broken |

More parameters consistently lose to the step-time cost. The 12L 3x sweet spot balances model capacity against training steps within the wallclock.

### Feature stacking

| Addition | BPB | Δ |
|----------|-----|---|
| Base | 1.1570 | — |
| + XSA last 4 | 1.1532 | -0.004 |
| + Partial RoPE 16 | 1.1525 | -0.005 |
| + Both | **1.1502** | **-0.007** |

### Paper-faithful nGPT

I tested whether following the original paper more closely would help. It didn't — at this training length.

| Change | BPB | Δ |
|--------|-----|---|
| Remove `.abs()` from alpha (allow negative) | 1.3458 | +0.189 |
| Add s_z output scaling params | Crashed | — |

The paper allows negative alpha for "anti-learning" interpolation, but 5000 steps isn't enough for the optimizer to discover which dimensions benefit from it. The extra s_z parameters spread the optimization budget too thin.

### TTT (negative result)

Test-time training produces NaN on nGPT with renorm dequantization, regardless of learning rate (tested 5e-5 to 2e-3), with or without weight renormalization after TTT steps, even with 10/12 blocks frozen. The renorm dequant computational graph is too numerically fragile for gradient-based weight updates. Standard GPTQ (without renorm) tolerates TTT at very low LR — the renorm step is what breaks it.

## Architecture

```
Embeddings → BigramHash(8192) → 12× nGPT Blocks → Tied Head
                                      │
                               normalize(input)
                               attention(h_norm)
                               normalize(attn_out)
                               slerp(h_norm, attn_norm, α_a)
                               mlp(x_out)
                               normalize(mlp_out)
                               slerp(x_out, mlp_norm, α_m)
```

12 layers, 512d, GQA 8/4, 3x MLP with LeakyReLU(0.5)², U-Net skips, XSA on last 4 layers, Partial RoPE 16/64, full nGPT normalization on both sides of interpolation, forward-pass weight normalization. 30M params total.

Training: Muon (matrix_lr=0.025, WD=0.04, momentum=0.99) + AdamW for non-matrix params. 786K token batches, seq_len=2048, 20-step warmup, 3500-iter warmdown with SWA. Int6 QAT with STE, Full GPTQ, renorm dequant, zstd-22 compression, adaptive Hessian-weighted pruning (~7.8%).

## Interpretability

Logit lens and layer knockout analysis on the 11L standard model (used to guide RYS target selection):

| Layer | Logit lens BPB | Knockout Δ | Role |
|-------|---------------|------------|------|
| 0 | 5.46 (-4.88) | +2.31 | Token patterns (critical) |
| 1-2 | 3.82 | +0.58-0.65 | Encoder |
| 3-5 | 5.29 (+1.47) | +0.15-0.29 | U-Net transition (BPB increases) |
| 6-7 | 2.57 (-2.72) | +0.18-0.19 | "Thinking" layers |
| 8-10 | 1.19 (-1.38) | +0.14-0.16 | Prediction refinement |

CKA similarity reveals three circuits: Encoder (L0-2, mutual CKA 0.83), Transition (L3-6), and Prediction (L7-10, CKA 0.88). Layers 7-8-9 are the tightest functional cluster in the network.

## What didn't work

| Idea | Outcome | Takeaway |
|------|---------|----------|
| Riemannian Muon (tangent-plane projection) | 0.19 BPB behind standard Muon | Convergence speed matters more than geometric correctness at short training |
| Triton fused normalize kernel | 17% slower | Kernel launch overhead dominates at dim=512; PyTorch's vectorized ops are already fast enough |
| Removing weight normalization | +1.6 BPB | Renorm dequant is useless without unit-norm weights |
| Scaling up (wider MLP, more layers) | All worse | Extra step-time cost outweighs per-step quality |
| Paper-faithful alpha / s_z | +0.189 / crashed | Extra DOF needs training time we don't have |
| TTT on nGPT + renorm dequant | NaN | Renorm dequant graph too fragile for gradients |

## Gap to SOTA

1.1502 is 0.031 behind SOTA (1.1194). The gap comes from two things: nGPT runs at 121ms/step vs the standard model's 88ms (fewer training steps), and TTT is broken on renorm-dequantized weights (SOTA uses TTT for ~-0.002). Both are fundamental costs of the hypersphere architecture under these constraints, not something I could optimize away.

The point of this PR isn't the BPB number — it's the findings. nGPT was written off by this competition after one failed attempt, but the failure was three trivial init bugs. The architecture actually works, compresses well, and has interesting properties (the compile precision bug, the compression paradox, the quantization-renormalization interaction). I hope this is useful to anyone thinking about constrained-norm architectures for small models.

## References

- [nGPT: Normalized Transformer with Representation Learning on the Hypersphere](https://arxiv.org/abs/2410.01131) (Loshchilov et al., ICLR 2025)
- [PR #831: Why Novel Architectures Fail at 16MB](https://github.com/openai/parameter-golf/pull/831)
- [PR #579: The Frugendorff](https://github.com/openai/parameter-golf/pull/579) (weight sharing)
- [PyTorch Issue #168126](https://github.com/pytorch/pytorch/issues/168126) (compile precision divergence)
