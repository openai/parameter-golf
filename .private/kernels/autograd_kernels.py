import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ==============================================================================
# TARGET #1: Fused resid_mix + RMSNorm (Priority 1)
# ==============================================================================

@triton.jit
def resid_mix_rmsnorm_fwd_kernel(
    x_ptr, x0_ptr, mix_ptr, n_ptr, rstd_ptr,
    stride_x_m, stride_x_k,
    stride_x0_m, stride_x0_k,
    stride_n_m, stride_n_k,
    stride_mix_0, stride_mix_1,
    K, eps,
    BLOCK_K: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_K)
    mask = col_offsets < K

    x_ptrs = x_ptr + row_idx * stride_x_m + col_offsets * stride_x_k
    x0_ptrs = x0_ptr + row_idx * stride_x0_m + col_offsets * stride_x0_k

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x0 = tl.load(x0_ptrs, mask=mask, other=0.0)

    mix0_ptrs = mix_ptr + col_offsets * stride_mix_1
    mix1_ptrs = mix_ptr + stride_mix_0 + col_offsets * stride_mix_1

    m0 = tl.load(mix0_ptrs, mask=mask, other=0.0)
    m1 = tl.load(mix1_ptrs, mask=mask, other=0.0)

    z = m0 * x + m1 * x0
    z_sq = z * z
    variance = tl.sum(z_sq, axis=0) / K
    rstd = tl.math.rsqrt(variance + eps)

    n = z * rstd

    n_ptrs = n_ptr + row_idx * stride_n_m + col_offsets * stride_n_k
    tl.store(n_ptrs, n, mask=mask)
    tl.store(rstd_ptr + row_idx, rstd)

@triton.jit
def resid_mix_rmsnorm_bwd_kernel(
    dn_ptr, x_ptr, x0_ptr, mix_ptr, rstd_ptr,
    dx_ptr, dx0_ptr, dz_ptr,
    stride_dn_m, stride_x_m, stride_x0_m,
    stride_mix_0, stride_mix_1,
    stride_dx_m, stride_dx0_m, stride_dz_m,
    K,
    BLOCK_K: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_K)
    mask = col_offsets < K

    dn_ptrs = dn_ptr + row_idx * stride_dn_m + col_offsets
    dn = tl.load(dn_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_ptrs = x_ptr + row_idx * stride_x_m + col_offsets
    x0_ptrs = x0_ptr + row_idx * stride_x0_m + col_offsets

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x0 = tl.load(x0_ptrs, mask=mask, other=0.0).to(tl.float32)

    mix0_ptrs = mix_ptr + col_offsets * stride_mix_1
    mix1_ptrs = mix_ptr + stride_mix_0 + col_offsets * stride_mix_1

    m0 = tl.load(mix0_ptrs, mask=mask, other=0.0).to(tl.float32)
    m1 = tl.load(mix1_ptrs, mask=mask, other=0.0).to(tl.float32)

    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)

    z = m0 * x + m1 * x0
    n = z * rstd

    mean_dn_n = tl.sum(dn * n, axis=0) / K
    dz = rstd * (dn - n * mean_dn_n)

    dx = dz * m0
    dx0 = dz * m1

    dx_ptrs = dx_ptr + row_idx * stride_dx_m + col_offsets
    dx0_ptrs = dx0_ptr + row_idx * stride_dx0_m + col_offsets
    dz_ptrs = dz_ptr + row_idx * stride_dz_m + col_offsets

    tl.store(dx_ptrs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dx0_ptrs, dx0.to(dx0_ptr.dtype.element_ty), mask=mask)
    tl.store(dz_ptrs, dz.to(dz_ptr.dtype.element_ty), mask=mask)


class FusedResidMixRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x0, mix, eps=1e-6):
        M, K = x.shape
        n = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_K = triton.next_power_of_2(K)
        grid = (M, )

        resid_mix_rmsnorm_fwd_kernel[grid](
            x, x0, mix, n, rstd,
            x.stride(0), x.stride(1),
            x0.stride(0), x0.stride(1),
            n.stride(0), n.stride(1),
            mix.stride(0), mix.stride(1),
            K, eps,
            BLOCK_K=BLOCK_K
        )

        ctx.save_for_backward(x, x0, mix, rstd)
        return n

    @staticmethod
    def backward(ctx, grad_output):
        x, x0, mix, rstd = ctx.saved_tensors
        M, K = x.shape

        dx = torch.empty_like(x)
        dx0 = torch.empty_like(x0)
        dz = torch.empty_like(x)

        BLOCK_K = triton.next_power_of_2(K)
        grid = (M, )

        resid_mix_rmsnorm_bwd_kernel[grid](
            grad_output, x, x0, mix, rstd,
            dx, dx0, dz,
            grad_output.stride(0), x.stride(0), x0.stride(0),
            mix.stride(0), mix.stride(1),
            dx.stride(0), dx0.stride(0), dz.stride(0),
            K,
            BLOCK_K=BLOCK_K
        )

        dmix_0 = torch.sum(dz * x, dim=0)
        dmix_1 = torch.sum(dz * x0, dim=0)
        dmix = torch.stack([dmix_0, dmix_1], dim=0)

        return dx, dx0, dmix, None


def fused_resid_mix_rmsnorm(x, x0, mix, eps=1e-6):
    """Drop-in replacement for: rms_norm(mix[0]*x + mix[1]*x0)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x0_2d = x0.reshape(-1, x0.shape[-1]).contiguous()
    mix_c = mix.contiguous()
    result = FusedResidMixRMSNormFunction.apply(x_2d, x0_2d, mix_c, eps)
    return result.reshape(orig_shape)


# ==============================================================================
# Test
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    M, K = 4096, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x0 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    mix = torch.randn(2, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    # Reference
    mix_ref = mix.detach().clone().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    x0_ref = x0.detach().clone().requires_grad_(True)
    z_ref = mix_ref[0] * x_ref + mix_ref[1] * x0_ref
    n_ref = F.rms_norm(z_ref, (K,))
    loss_ref = n_ref.sum()
    loss_ref.backward()

    # Fused
    n_fused = fused_resid_mix_rmsnorm(x, x0, mix)
    loss_fused = n_fused.sum()
    loss_fused.backward()

    print(f"Forward max diff: {(n_ref - n_fused).abs().max().item():.6f}")
    print(f"dx max diff: {(x_ref.grad - x.grad).abs().max().item():.6f}")
    print(f"dx0 max diff: {(x0_ref.grad - x0.grad).abs().max().item():.6f}")
    print(f"dmix max diff: {(mix_ref.grad - mix.grad).abs().max().item():.6f}")

    # Benchmark
    import time
    def bench(fn, warmup=10, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    def ref_fn():
        z = mix[0] * x + mix[1] * x0
        n = F.rms_norm(z, (K,))
        n.sum().backward()

    def fused_fn():
        n = fused_resid_mix_rmsnorm(x, x0, mix)
        n.sum().backward()

    ref_ms = bench(ref_fn)
    fused_ms = bench(fused_fn)
    print(f"Reference: {ref_ms:.3f}ms, Fused: {fused_ms:.3f}ms, Speedup: {ref_ms/fused_ms:.2f}x")
