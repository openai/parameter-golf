# Kernel Research Brief: Autograd-Compatible Fused Triton Kernels for Parameter Golf

## The Problem

We have Triton kernels that are 1.26-1.75x faster than torch.compile for key operations, but they only work during eval (inference mode). Training requires autograd — the kernel outputs need gradient computation for backpropagation. Without this, the kernels can't speed up the 80ms/step training loop, which is the actual bottleneck.

## What We Need

Custom `torch.autograd.Function` wrappers for our Triton kernels that provide both forward AND backward passes. This lets PyTorch's autograd engine call our fast Triton kernels during training, not just eval.

## Target Kernel #1: Fused ReLU² MLP (1.26x speedup)

**Operation:** `y = proj(relu(fc(x))²)` — two matmuls with relu² activation between them.

**Shapes (Parameter Golf model):**
- Input: `x` is `[B*S, 512]` where B=batch, S=2048
- `fc.weight` is `[1536, 512]` (MLP 3x expansion)
- `proj.weight` is `[512, 1536]`
- Output: `y` is `[B*S, 512]`

**Forward:** `h = F.linear(x, fc_weight)` → `h_relu = relu(h)` → `h_sq = h_relu²` → `y = F.linear(h_sq, proj_weight)`

**Backward needs:**
- `dL/dx = dL/dy @ proj_weight @ diag(2*h_relu * (h > 0)) @ fc_weight`
- `dL/d_proj_weight = dL/dy.T @ h_sq`
- `dL/d_fc_weight = (dL/dy @ proj_weight * 2*h_relu * (h > 0)).T @ x`

**The fusion opportunity:** During forward, save `h` (pre-relu) in the context for backward. The backward then fuses: `grad_output @ proj_weight` (GEMM) → multiply by `2*relu(h)*(h>0)` (pointwise) → `result @ fc_weight` or `.T @ x` (GEMM). The pointwise relu²-derivative can be fused into either GEMM's epilogue.

**Reference:** Our Makora-generated forward kernel is at `.private/kernels/best_relu2_mlp_cuda_1.26x.py`. It fuses the relu² + second matmul. The backward needs a similar fusion.

## Target Kernel #2: Fused RMSNorm + Linear Projection (1.47x speedup)

**Operation:** `y = rms_norm(x) @ W.T` — normalization fused with matmul.

**Shapes:**
- Input: `x` is `[B*S, 512]`
- Weight: `W` is `[N, 512]` where N=512 (Q proj), 256 (K proj), or 256 (V proj)
- Output: `y` is `[B*S, N]`

**Forward:** `rstd = rsqrt(mean(x²) + eps)` → `x_norm = x * rstd` → `y = x_norm @ W.T`

**Backward needs:**
- `dL/dx` requires both the GEMM backward AND the RMSNorm backward
- Save `rstd` and `x` in context
- `dL/dx_norm = dL/dy @ W` (GEMM)
- `dL/dx = rstd * (dL/dx_norm - x_norm * mean(dL/dx_norm * x_norm))` (RMSNorm backward)
- `dL/dW = dL/dy.T @ x_norm` (weight gradient GEMM)

**The fusion opportunity:** The RMSNorm backward is a row-reduction + pointwise op. Fusing it with the GEMM backward eliminates a full HBM read/write of the intermediate `dL/dx_norm`.

**Reference:** Our kernel is at `.private/kernels/best_rmsnorm_qkv_triton_1.48x.py`.

## Target Kernel #3: Fused resid_mix + RMSNorm (1.08x, but called 9x per step)

**Operation:** `n = rms_norm(mix[0] * x + mix[1] * x0)` — weighted residual blend + normalization.

**Shapes:**
- `x`, `x0` are `[B*S, 512]`
- `mix` is `[2, 512]` (learned per-channel blending weights)
- Output: `n` is `[B*S, 512]`

**This is the simplest kernel to make autograd-compatible** because it's purely pointwise + reduction (no GEMM). The backward is straightforward chain rule through RMSNorm and the linear blend.

## How to Implement

Use `torch.autograd.Function`:

```python
class FusedReLU2MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fc_weight, proj_weight):
        # Run Triton forward kernel
        h = F.linear(x, fc_weight)  # or fused kernel
        h_relu = torch.relu(h)
        h_sq = h_relu * h_relu
        y = triton_fused_relu_sq_proj(h_sq, proj_weight)  # our 1.26x kernel
        ctx.save_for_backward(x, h, proj_weight, fc_weight)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, h, proj_weight, fc_weight = ctx.saved_tensors
        h_relu = torch.relu(h)
        relu_deriv = 2.0 * h_relu * (h > 0).float()

        # dL/d_proj_weight
        h_sq = h_relu * h_relu
        grad_proj = grad_output.t() @ h_sq

        # dL/dh (through proj + relu²)
        grad_h = (grad_output @ proj_weight) * relu_deriv

        # dL/d_fc_weight and dL/dx
        grad_fc = grad_h.t() @ x
        grad_x = grad_h @ fc_weight

        return grad_x, grad_fc, grad_proj
```

The above is the PyTorch reference. The Triton optimization is fusing the `(grad_output @ proj_weight) * relu_deriv` into a single kernel (GEMM with pointwise epilogue), and similarly for the weight gradient GEMMs.

## Constraints

- **Must produce identical results** to the PyTorch reference (within bf16 precision)
- **Must work with torch.compile** (the model is compiled with `fullgraph=True`)
- **Must handle bf16 compute with fp32 accumulation** (CastedLinear stores weights in fp32, casts to bf16 for matmul)
- Target hardware: NVIDIA H100 SXM 80GB
- Problem sizes: batch*seq = 8192-16384, dim = 512, hidden = 1536

## Expected Impact

At 80ms/step with 9 layers:
- Each block's MLP forward+backward is ~25% of step time (~20ms)
- 1.26x speedup on MLP = ~4ms saved per step
- 9 blocks = ~36ms saved? No — torch.compile already fuses some of this
- Realistic: 5-10ms/step savings = 6-12% speedup = ~500-900 more training steps
- At this stage, 500 extra steps ≈ 0.005-0.01 bpb

## Files

- `.private/kernels/best_relu2_mlp_cuda_1.26x.py` — Makora forward kernel
- `.private/kernels/best_rmsnorm_qkv_triton_1.48x.py` — Makora RMSNorm+QKV forward kernel
- `.private/kernels/best_softcap_ce_cuda_1.70x.py` — Makora softcap+CE forward kernel
- Our Triton skills: `.agents/skills/triton-kernels/` — reference patterns for fused kernels

## Priority Order

1. **Fused resid_mix + RMSNorm** (simplest, no GEMM backward)
2. **Fused ReLU² MLP** (highest absolute savings)
3. **Fused RMSNorm + Linear** (complex backward, highest Makora speedup)
