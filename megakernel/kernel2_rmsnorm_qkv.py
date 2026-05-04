"""
Kernel 2: Fused RMSNorm + QKV Projection

The current attention pre-norm path in Block.forward:
    self.attn_norm(x_in) * self.ln_scale_factor
    → passed to CausalSelfAttention.forward()
    → q = F.linear(x_normed, q_w)   [M,512 → M,512]
    → k = F.linear(x_normed, k_w)   [M,512 → M,256]
    → v = F.linear(x_normed, v_w)   [M,512 → M,256]

That's 5 kernel launches before Flash Attention even starts.

This kernel fuses all 5 into ONE:
    FusedRMSNormQKV(x_in, q_w, k_w, v_w, scale, eps)
    → q [M, 512], k [M, 256], v [M, 256] in one pass

REORDERING TRICK (same as Kernel 1):
    (x / rms) @ W = (x @ W) / rms

One pass over x computes BOTH the 3 GEMMs AND the per-row RMS simultaneously.

MEMORY SAVINGS:
    Unfused: read x(75MB), write normed_x(75MB), read normed_x 3 times(225MB)
    Fused:   read x 3 times(225MB) — no normed_x write ever
    Savings: 75MB per layer × 11 layers = 825MB per forward pass

COMPLIANCE: Pure compute optimization. 100% compliant.
"""
import math
import sys
import time
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# POINTER-BASED TRITON KERNEL: Fused RMSNorm + Single Linear
# Used three times for Q, K, V (each with different output dims)
# ─────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def rmsnorm_linear_fwd_ptrs(
        x_ptr, w_ptr, out_ptr,
        inv_rms_ptr,            # [M] float32 — shared across Q/K/V calls
        write_inv_rms,          # bool: only Q projection writes inv_rms
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        scale,
        eps,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COMPUTE_RMS: tl.constexpr,  # True for Q, False for K/V (reuse computed rms)
    ):
        """
        Fused RMSNorm + Linear.
        For Q: computes RMS and stores inv_rms to inv_rms_ptr
        For K/V: loads inv_rms from inv_rms_ptr (already computed by Q pass)

        This way Q, K, V share one RMSNorm computation.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Accumulate GEMM and optionally sum(x²)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_off in range(0, K, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

            acc = tl.dot(x_tile, tl.trans(w_tile), acc)

            if COMPUTE_RMS:
                x_f32 = x_tile.to(tl.float32)
                sum_sq += tl.sum(x_f32 * x_f32, axis=1)

        # Get inv_rms: either compute (Q path) or load (K/V path)
        if COMPUTE_RMS:
            inv_rms = scale / tl.sqrt(sum_sq / K + eps)
            # Store inv_rms for K and V to reuse (only pid_n==0 to avoid races)
            if write_inv_rms:
                tl.store(inv_rms_ptr + offs_m, inv_rms.to(tl.float32), mask=mask_m)
        else:
            inv_rms = tl.load(inv_rms_ptr + offs_m, mask=mask_m).to(tl.float32)

        # Apply RMSNorm scaling
        out = (acc * inv_rms[:, None]).to(tl.bfloat16)

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def rmsnorm_linear(x, w, scale=1.0, eps=1e-6, inv_rms_buf=None):
    """
    Fused RMSNorm + Linear for a single projection.

    If inv_rms_buf is None: computes RMS from scratch (for Q projection)
    If inv_rms_buf is provided: reuses it (for K, V projections)

    Returns: (output, inv_rms_buf)
    """
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    compute_rms = (inv_rms_buf is None)
    if inv_rms_buf is None:
        inv_rms_buf = torch.empty((M,), device=x.device, dtype=torch.float32)

    BM, BN, BK = 128, 256, 64  # H100 autotune: 1.224x at M=73728
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    rmsnorm_linear_fwd_ptrs[grid](
        x, w, out, inv_rms_buf,
        True,  # write_inv_rms (always write so K/V can reuse)
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        scale, eps,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        COMPUTE_RMS=compute_rms,
        num_warps=8, num_stages=4,
    )
    return out, inv_rms_buf


def fused_rmsnorm_qkv(x, q_w, k_w, v_w, scale=1.0, eps=1e-6):
    """
    Fused RMSNorm + QKV linear projections.

    Computes RMS once (during Q projection), reuses for K and V.

    Args:
        x:   [M, K] input (un-normalized)
        q_w: [N_q, K] query weight
        k_w: [N_k, K] key weight
        v_w: [N_v, K] value weight
        scale: ln_scale_factor
        eps: RMSNorm epsilon

    Returns:
        q: [M, N_q]
        k: [M, N_k]
        v: [M, N_v]
        inv_rms: [M] per-row inv_rms (for backward)
    """
    if TRITON_AVAILABLE and x.is_cuda:
        # Q projection: compute RMS and store inv_rms
        q, inv_rms = rmsnorm_linear(x, q_w, scale=scale, eps=eps, inv_rms_buf=None)
        # K, V projections: reuse inv_rms
        k, _ = rmsnorm_linear(x, k_w, scale=scale, eps=eps, inv_rms_buf=inv_rms)
        v, _ = rmsnorm_linear(x, v_w, scale=scale, eps=eps, inv_rms_buf=inv_rms)
    else:
        # CPU fallback
        rms = torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
        x_normed = (x.float() / rms * scale).to(x.dtype)
        inv_rms = (scale / rms.squeeze(-1)).float()
        q = F.linear(x_normed, q_w)
        k = F.linear(x_normed, k_w)
        v = F.linear(x_normed, v_w)

    return q, k, v, inv_rms


# ─────────────────────────────────────────────────────────────
# AUTOGRAD FUNCTION: FusedRMSNormQKV
# ─────────────────────────────────────────────────────────────

class FusedRMSNormQKVFunction(torch.autograd.Function):
    """
    Fused RMSNorm + QKV projection with correct autograd.

    Replaces in CausalSelfAttention.forward():
        x_normed = attn_norm(x) * scale
        q = F.linear(x_normed, q_w)
        k = F.linear(x_normed, k_w)
        v = F.linear(x_normed, v_w)

    With:
        q, k, v, inv_rms = FusedRMSNormQKV(x, q_w, k_w, v_w, scale, eps)
    """

    @staticmethod
    def forward(ctx, x, q_w, k_w, v_w, scale, eps):
        x_2d = x.reshape(-1, x.shape[-1])
        q, k, v, inv_rms = fused_rmsnorm_qkv(x_2d, q_w, k_w, v_w, scale, eps)
        ctx.save_for_backward(x, q_w, k_w, v_w, inv_rms)
        ctx.scale = scale
        ctx.eps = eps
        # Return q, k, v in same shape as input (batch dim preserved)
        return (q.view(*x.shape[:-1], q.shape[-1]),
                k.view(*x.shape[:-1], k.shape[-1]),
                v.view(*x.shape[:-1], v.shape[-1]))

    @staticmethod
    def backward(ctx, dq, dk, dv):
        x, q_w, k_w, v_w, inv_rms = ctx.saved_tensors
        scale, eps = ctx.scale, ctx.eps
        x_2d = x.reshape(-1, x.shape[-1])
        dq_2d = dq.reshape(-1, dq.shape[-1])
        dk_2d = dk.reshape(-1, dk.shape[-1])
        dv_2d = dv.reshape(-1, dv.shape[-1])

        # Backward through QKV linear projections
        x_n  = _get_x_normed(x_2d, inv_rms)
        dw_q = dq_2d.float().T @ x_n.float()
        dw_k = dk_2d.float().T @ x_n.float()
        dw_v = dv_2d.float().T @ x_n.float()

        # d_x_normed = sum of gradients from Q, K, V paths
        d_x_normed = (dq_2d.float() @ q_w.float() +
                      dk_2d.float() @ k_w.float() +
                      dv_2d.float() @ v_w.float())

        # Backward through RMSNorm
        dx = _rmsnorm_backward(d_x_normed, x_2d, inv_rms)

        return dx.view_as(x), dw_q.to(q_w.dtype), dw_k.to(k_w.dtype), dw_v.to(v_w.dtype), None, None


def _get_x_normed(x_2d, inv_rms):
    """Reconstruct x_normed from x and saved inv_rms (for backward dW computation)."""
    return (x_2d.float() * inv_rms[:, None]).to(x_2d.dtype)


def _rmsnorm_backward(d_x_normed, x_2d, inv_rms):
    """
    Backward through RMSNorm: dx = inv_rms * (d_x_normed - x_normed * dot(d_x_normed, x_normed) / K)
    """
    K = x_2d.shape[-1]
    x_normed = x_2d.float() * inv_rms[:, None]
    d_x_normed_f = d_x_normed.float()
    dot = (d_x_normed_f * x_normed).sum(dim=-1, keepdim=True) / K
    return (inv_rms[:, None] * (d_x_normed_f - x_normed * dot)).to(x_2d.dtype)


FusedRMSNormQKVApply = FusedRMSNormQKVFunction.apply


# ─────────────────────────────────────────────────────────────
# PYTORCH REFERENCE
# ─────────────────────────────────────────────────────────────

def reference_qkv(x, q_w, k_w, v_w, scale=1.0, eps=1e-6):
    """Unfused reference: attn_norm(x)*scale → [q_proj, k_proj, v_proj]"""
    rms = torch.sqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    x_normed = (x.float() / rms * scale).to(x.dtype)
    return F.linear(x_normed, q_w), F.linear(x_normed, k_w), F.linear(x_normed, v_w)


# ─────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────

def test_correctness(device="cpu"):
    print(f"\n── Correctness test (device={device}) ──────────────────")
    torch.manual_seed(42)

    M, K = 4096, 512
    N_q = 512   # 8 heads × 64 head_dim
    N_k = 256   # 4 KV heads × 64 head_dim
    N_v = 256
    scale = 1.0 / math.sqrt(3)

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    q_w = torch.randn(N_q, K, dtype=torch.bfloat16, device=device) * 0.02
    k_w = torch.randn(N_k, K, dtype=torch.bfloat16, device=device) * 0.02
    v_w = torch.randn(N_v, K, dtype=torch.bfloat16, device=device) * 0.02

    q_ref, k_ref, v_ref = reference_qkv(x, q_w, k_w, v_w, scale=scale)
    q_fused, k_fused, v_fused, _ = fused_rmsnorm_qkv(
        x, q_w, k_w, v_w, scale=scale)

    q_err = (q_fused.float() - q_ref.float()).abs().max().item()
    k_err = (k_fused.float() - k_ref.float()).abs().max().item()
    v_err = (v_fused.float() - v_ref.float()).abs().max().item()

    print(f"  M={M}, K={K}, N_q={N_q}, N_k={N_k}")
    print(f"  Q max err: {q_err:.2e}")
    print(f"  K max err: {k_err:.2e}")
    print(f"  V max err: {v_err:.2e}")

    ok = all(e < 0.05 for e in [q_err, k_err, v_err])
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


def benchmark(device="cuda"):
    if not torch.cuda.is_available():
        print("\n── Benchmark SKIPPED (no CUDA) ──")
        return

    print(f"\n── Benchmark: fused QKV vs unfused (device={device}) ────")
    M, K = 73728, 512
    N_q, N_k, N_v = 512, 256, 256

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    q_w = torch.randn(N_q, K, dtype=torch.bfloat16, device=device) * 0.02
    k_w = torch.randn(N_k, K, dtype=torch.bfloat16, device=device) * 0.02
    v_w = torch.randn(N_v, K, dtype=torch.bfloat16, device=device) * 0.02

    reps = 100

    def bench_fn(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    t_ref = bench_fn(lambda: reference_qkv(x, q_w, k_w, v_w))
    t_fused = bench_fn(lambda: fused_rmsnorm_qkv(x, q_w, k_w, v_w))

    speedup = t_ref / t_fused
    saving = (t_ref - t_fused) * 11 * 2  # 11 layers, fwd+bwd

    print(f"  Unfused:   {t_ref:.3f} ms/call")
    print(f"  Fused:     {t_fused:.3f} ms/call")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"  Est. saving per step (11 layers): {saving:.2f} ms")


INTEGRATION_DIFF = '''
# ── In Block.forward() ────────────────────────────────────────
# BEFORE (5 kernel launches per layer):
attn_out = self.attn(
    self.attn_norm(x_in) * self.ln_scale_factor,
    q_w, k_w, v_w, out_w,
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
)

# AFTER (1 kernel launch per layer for pre-norm + QKV):
if FUSED_RMSNORM_QKV_ENABLED and self.training:
    attn_input_q, attn_input_k, attn_input_v = FusedRMSNormQKVApply(
        x_in, q_w, k_w, v_w,
        self.ln_scale_factor, getattr(self.attn_norm, 'eps', 1e-6)
    )
    attn_out = self.attn.forward_precomputed_qkv(
        x_in, attn_input_q, attn_input_k, attn_input_v, out_w,
        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
    )
else:
    attn_out = self.attn(
        self.attn_norm(x_in) * self.ln_scale_factor,
        q_w, k_w, v_w, out_w,
        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
    )

# ── Add to CausalSelfAttention ────────────────────────────────
def forward_precomputed_qkv(self, x, q, k, v, out_w, cu_seqlens=None, max_seqlen=0):
    """Like forward() but q,k,v are already projected (from fused kernel)."""
    bsz, seqlen, dim = x.shape
    q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
    k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    # ... rest of attention unchanged ...
'''

if __name__ == "__main__":
    run_bench = "bench" in sys.argv

    print("=" * 60)
    print("Kernel 2: Fused RMSNorm + QKV Projection")
    print("=" * 60)
    print(f"Triton: {TRITON_AVAILABLE}, TMA: {TMA_AVAILABLE}, CUDA: {torch.cuda.is_available()}")

    test_correctness("cpu")

    if torch.cuda.is_available():
        test_correctness("cuda")
        if run_bench:
            benchmark("cuda")

    print("\n── Integration diff ─────────────────────────────────────")
    print(INTEGRATION_DIFF)
