"""
Fused RMSNorm + Linear projection kernel for Parameter Golf.

Computes: y = rms_norm(x) @ W^T
Where rms_norm(x) = x * rsqrt(mean(x^2) + eps)

Fuses the normalization into the GEMM prologue so the normalized
tensor never hits HBM. Each tile of input rows is normalized in
shared memory / registers before the dot product.

For the QKV case, we concatenate W_q, W_k, W_v and do a single
fused matmul, then split the output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

if _HAS_TRITON:
    @triton.jit
    def _rmsnorm_linear_kernel(
        x_ptr, w_ptr, out_ptr,
        M, K: tl.constexpr, N,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused RMSNorm + Linear: out[m, n] = rms_norm(x[m, :]) @ W[n, :].T"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Row and column offsets for this tile
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        m_mask = rm < M
        n_mask = rn < N

        # === Phase 1: Compute RMS norm statistics for this tile's rows ===
        # Accumulate sum of squares across K dimension
        ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + rk
            k_mask = k_offs < K
            x_tile = tl.load(
                x_ptr + rm[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ss += tl.sum(x_tile * x_tile, axis=1)

        # rsqrt(mean(x^2) + eps)
        rstd = tl.math.rsqrt(ss / K + eps)  # (BLOCK_M,)

        # === Phase 2: Fused normalized matmul ===
        # Accumulate out[m, n] = sum_k(x[m, k] * rstd[m] * W[n, k])
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + rk
            k_mask = k_offs < K

            # Load x tile and normalize in-register
            x_tile = tl.load(
                x_ptr + rm[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_normed = x_tile * rstd[:, None]  # Apply RMSNorm per-row

            # Load weight tile (W is [N, K], we want W[n, k])
            w_tile = tl.load(
                w_ptr + rn[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Matmul: x_normed[M, K] @ W[N, K].T = x_normed[M, K] @ W.T[K, N]
            acc += tl.dot(x_normed.to(tl.bfloat16), tl.trans(w_tile.to(tl.bfloat16)))

        # Store output
        out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def fused_rmsnorm_linear(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Compute rms_norm(x) @ weight.T in a single fused kernel.

    Args:
        x: [*, K] input tensor (bfloat16)
        weight: [N, K] weight matrix (any dtype, cast to bf16)
        eps: RMSNorm epsilon

    Returns:
        out: [*, N] output tensor (bfloat16)
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    w = weight.to(torch.bfloat16).contiguous()
    M, K = x_2d.shape
    N = w.shape[0]

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    # Grid: one program per (BLOCK_M rows, BLOCK_N columns) tile
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = min(K, 128)  # Process K dimension in chunks

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _rmsnorm_linear_kernel[grid](
        x_2d, w, out,
        M, K, N,
        x_2d.stride(0), x_2d.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out.reshape(*orig_shape[:-1], N)


def fused_rmsnorm_qkv(
    x: torch.Tensor,
    w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute q, k, v = rms_norm(x) @ [W_q, W_k, W_v].T in one fused call.

    Concatenates weights, does one fused kernel call, splits output.
    """
    w_qkv = torch.cat([w_q, w_k, w_v], dim=0)  # [Nq+Nk+Nv, K]
    qkv = fused_rmsnorm_linear(x, w_qkv, eps)
    Nq = w_q.shape[0]
    Nk = w_k.shape[0]
    return qkv[..., :Nq], qkv[..., Nq:Nq+Nk], qkv[..., Nq+Nk:]


# === Test ===
if __name__ == "__main__":
    torch.manual_seed(42)
    B, S, D = 4, 1024, 512
    Nq, Nk = 512, 256

    x = torch.randn(B, S, D, dtype=torch.bfloat16, device="cuda")
    w_q = torch.randn(Nq, D, dtype=torch.bfloat16, device="cuda")
    w_k = torch.randn(Nk, D, dtype=torch.bfloat16, device="cuda")
    w_v = torch.randn(Nk, D, dtype=torch.bfloat16, device="cuda")

    # Reference
    n = F.rms_norm(x, (D,))
    ref_q = F.linear(n, w_q)
    ref_k = F.linear(n, w_k)
    ref_v = F.linear(n, w_v)

    # Fused
    fq, fk, fv = fused_rmsnorm_qkv(x, w_q, w_k, w_v)

    print(f"Q max diff: {(ref_q - fq).abs().max().item():.6f}")
    print(f"K max diff: {(ref_k - fk).abs().max().item():.6f}")
    print(f"V max diff: {(ref_v - fv).abs().max().item():.6f}")

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

    ref_ms = bench(lambda: (F.linear(F.rms_norm(x, (D,)), w_q), F.linear(F.rms_norm(x, (D,)), w_k), F.linear(F.rms_norm(x, (D,)), w_v)))
    fused_ms = bench(lambda: fused_rmsnorm_qkv(x, w_q, w_k, w_v))
    print(f"Reference: {ref_ms:.3f}ms, Fused: {fused_ms:.3f}ms, Speedup: {ref_ms/fused_ms:.2f}x")
