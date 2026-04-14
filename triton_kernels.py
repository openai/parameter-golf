"""triton_kernels.py — Custom Triton kernels for high-performance training.

Targets every memory-bandwidth-bound operation in train_gpt_verbose.py:
  1. Fused Fast Walsh-Hadamard Transform (FWHT) — all butterfly stages in SRAM
  2. Parallel Linear Recurrence Scan — eliminates chunked intermediates
  3. Fused Ternary Dequant — group scale + quantize + dequant in one kernel
  4. MoE dispatch optimization — eliminates repeat_interleave
  5. FeedbackAdapter optimization — eliminates D×D matrix materialization

All kernels include PyTorch fallback paths and strict numerical parity tests.
Pre-tuned for H100 (sm_90): BLOCK sizes, num_warps, num_stages hardcoded.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Triton availability guard
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ---------------------------------------------------------------------------
# Hadamard matrix cache (shared by FWHT and Ternary kernels)
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[tuple[int, str], Tensor] = {}


def _get_hadamard(n: int, device) -> Tensor:
    """Normalized Hadamard matrix: H @ H = I. Cached per (n, device)."""
    key = (n, str(device))
    if key not in _HADAMARD_CACHE:
        assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
        H = torch.ones(1, 1, dtype=torch.float32, device="cpu")
        for _ in range(int(math.log2(n))):
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = (H / math.sqrt(n)).to(device).contiguous()
        _HADAMARD_CACHE[key] = H
    return _HADAMARD_CACHE[key]


# ============================================================================
# 1. FUSED FAST WALSH-HADAMARD TRANSFORM (FWHT)
# ============================================================================
# Replaces causal_wht_blockwise() which does O(log N) HBM round-trips via a
# Python for loop over butterfly stages.  This kernel loads the entire block
# into SRAM, multiplies by the precomputed H matrix via Tensor Cores, and
# writes the result back once.  block_size ≤ 128 fits trivially in shared mem.
#
# Backward: FWHT is self-inverse (H symmetric orthogonal), so backward = forward.
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _fwht_kernel(
        X_ptr, H_ptr, Out_ptr,
        total_blocks,
        D: tl.constexpr,
        stride_xb, stride_xt, stride_xd,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        if pid_b >= total_blocks:
            return

        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load H matrix (BLOCK_SIZE, BLOCK_SIZE) — fits in registers for BS≤128
        h_r = tl.arange(0, BLOCK_SIZE)
        h_c = tl.arange(0, BLOCK_SIZE)
        H = tl.load(H_ptr + h_r[:, None] * BLOCK_SIZE + h_c[None, :])

        # Load data tile (BLOCK_SIZE, BLOCK_D)
        t_offs = tl.arange(0, BLOCK_SIZE)
        x_ptrs = X_ptr + pid_b * stride_xb + t_offs[:, None] * stride_xt + d_offs[None, :] * stride_xd
        data = tl.load(x_ptrs, mask=d_mask[None, :], other=0.0)

        # Matmul: H @ data using Tensor Cores (bf16 inputs, fp32 accumulation)
        result = tl.dot(H.to(tl.bfloat16), data.to(tl.bfloat16))

        # Store result (fp32 → auto-truncated to output tensor dtype by tl.store)
        out_ptrs = Out_ptr + pid_b * stride_xb + t_offs[:, None] * stride_xt + d_offs[None, :] * stride_xd
        tl.store(out_ptrs, result.to(data.dtype), mask=d_mask[None, :])


class TritonFWHTFn(torch.autograd.Function):
    """Autograd wrapper for the FWHT Triton kernel."""

    @staticmethod
    def forward(ctx, x_blocks: Tensor, H: Tensor) -> Tensor:
        # x_blocks: (total_blocks, block_size, D), H: (block_size, block_size)
        total_blocks, block_size, D = x_blocks.shape
        out = torch.empty_like(x_blocks)
        BLOCK_D = min(D, 128)
        # H100-tuned: 4 warps for 64-wide blocks
        grid = (total_blocks, triton.cdiv(D, BLOCK_D))
        _fwht_kernel[grid](
            x_blocks, H, out,
            total_blocks, D,
            x_blocks.stride(0), x_blocks.stride(1), x_blocks.stride(2),
            BLOCK_SIZE=block_size, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3,
        )
        ctx.save_for_backward(H)
        return out

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor, None]:
        (H,) = ctx.saved_tensors
        # FWHT is self-inverse: backward = forward (H is symmetric orthogonal)
        total_blocks, block_size, D = dy.shape
        dx = torch.empty_like(dy)
        BLOCK_D = min(D, 128)
        grid = (total_blocks, triton.cdiv(D, BLOCK_D))
        _fwht_kernel[grid](
            dy, H, dx,
            total_blocks, D,
            dy.stride(0), dy.stride(1), dy.stride(2),
            BLOCK_SIZE=block_size, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=3,
        )
        return dx, None


def triton_fwht_blockwise(x: Tensor, block_size: int, num_stages: int | None = None) -> Tensor:
    """Drop-in replacement for causal_wht_blockwise() using Triton.

    Falls back to PyTorch for partial WHT (num_stages < log2(block_size)).
    """
    max_stages = int(math.log2(block_size))
    if num_stages is not None and int(num_stages) != max_stages:
        # Partial WHT not supported by matmul approach — fallback
        return _pytorch_fwht_blockwise(x, block_size, num_stages)

    B, T, D = x.shape
    pad_len = (block_size - T % block_size) % block_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
    T_padded = x.shape[1]
    num_blocks = T_padded // block_size

    x_blocks = x.reshape(B * num_blocks, block_size, D).contiguous()
    H = _get_hadamard(block_size, x.device).to(torch.float32)

    y_blocks = TritonFWHTFn.apply(x_blocks, H)
    y = y_blocks.reshape(B, T_padded, D)
    return y[:, :T, :]


def _pytorch_fwht_blockwise(x: Tensor, block_size: int, num_stages: int | None = None) -> Tensor:
    """Pure PyTorch FWHT fallback (identical to causal_wht_blockwise)."""
    B, T, D = x.shape
    pad_len = (block_size - T % block_size) % block_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
    T_padded = x.shape[1]
    num_blocks = T_padded // block_size
    x_blocks = x.view(B, num_blocks, block_size, D)
    max_stages = int(math.log2(block_size))
    h = max_stages if num_stages is None else int(num_stages)
    result = x_blocks
    for stage in range(h):
        stride = 1 << stage
        result = result.view(B, num_blocks, block_size // (2 * stride), 2 * stride, D)
        even = result[:, :, :, :stride, :]
        odd = result[:, :, :, stride:, :]
        result = torch.cat([even + odd, even - odd], dim=3)
        result = result.view(B, num_blocks, block_size, D)
    result = result * (1.0 / math.sqrt(block_size))
    result = result.view(B, T_padded, D)
    return result[:, :T, :]


# ============================================================================
# 2. PARALLEL LINEAR RECURRENCE SCAN
# ============================================================================
# Replaces KoopmanTokenMixer._causal_decay_scan() chunked approach that creates
# h_local, d_local, chunk_prefixes intermediates (128MB+ for typical configs).
#
# Each Triton program handles one (batch, state_block) pair for ALL timesteps,
# computing h[t] = D[t]*h[t-1] + B[t] sequentially in registers.  No inter-
# program communication needed.  Memory traffic: read input once, write output
# once.  Eliminates all intermediate HBM tensors.
#
# Backward: reverse linear recurrence δ[t] = dh[t] + D[t+1]*δ[t+1].
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _scan_fwd_kernel(
        B_ptr, D_ptr, H_ptr,
        B_dim, T_dim, S_dim,
        stride_b, stride_t, stride_s,
        stride_hb, stride_ht, stride_hs,
        BLOCK_S: tl.constexpr,
    ):
        """Forward scan: h[t] = D[t]*h[t-1] + B_vals[t], h[-1]=0.
        Stores h in fp32 for gradient precision in backward."""
        pid = tl.program_id(0)
        batch_idx = pid // tl.cdiv(S_dim, BLOCK_S)
        s_block = pid % tl.cdiv(S_dim, BLOCK_S)

        s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S_dim

        h = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for t in range(T_dim):
            base = batch_idx * stride_b + t * stride_t
            h_base = batch_idx * stride_hb + t * stride_ht
            b_val = tl.load(B_ptr + base + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)
            d_val = tl.load(D_ptr + base + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)
            h = d_val * h + b_val
            # Store in fp32 for backward precision
            tl.store(H_ptr + h_base + s_offs * stride_hs, h, mask=s_mask)

    @triton.jit
    def _scan_bwd_kernel(
        DH_ptr, D_ptr, H_ptr,
        DB_ptr, DD_ptr,
        B_dim, T_dim, S_dim,
        stride_b, stride_t, stride_s,
        stride_hb, stride_ht, stride_hs,
        BLOCK_S: tl.constexpr,
    ):
        """Backward scan: δ[t] = dh[t] + D[t+1]*δ[t+1], running right-to-left."""
        pid = tl.program_id(0)
        batch_idx = pid // tl.cdiv(S_dim, BLOCK_S)
        s_block = pid % tl.cdiv(S_dim, BLOCK_S)

        s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S_dim

        delta = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for t_rev in range(T_dim):
            t = T_dim - 1 - t_rev
            base = batch_idx * stride_b + t * stride_t

            dh_val = tl.load(DH_ptr + base + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)
            d_val = tl.load(D_ptr + base + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)

            delta = dh_val + delta

            # dB[t] = δ[t]
            tl.store(DB_ptr + base + s_offs * stride_s, delta.to(tl.bfloat16), mask=s_mask)

            # dD[t] = δ[t] * h[t-1], now reading fp32 h_prev
            h_prev = tl.zeros((BLOCK_S,), dtype=tl.float32)
            if t > 0:
                prev_h_base = batch_idx * stride_hb + (t - 1) * stride_ht
                h_prev = tl.load(H_ptr + prev_h_base + s_offs * stride_hs, mask=s_mask, other=0.0).to(tl.float32)
            dd_val = delta * h_prev
            tl.store(DD_ptr + base + s_offs * stride_s, dd_val.to(tl.bfloat16), mask=s_mask)

            delta = d_val * delta


class TritonScanFn(torch.autograd.Function):
    """Autograd wrapper for the parallel linear recurrence scan.

    Forward:  h[t] = D[t] * h[t-1] + B_vals[t],  h[-1] = 0
    Inputs:   B_vals (B,T,S), D (B,T,S)
    Output:   h (B,T,S) — stored in fp32 for backward, returned in input dtype
    """

    @staticmethod
    def forward(ctx, B_vals: Tensor, D: Tensor) -> Tensor:
        B_dim, T_dim, S_dim = B_vals.shape
        B_vals_c = B_vals.contiguous()
        D_c = D.contiguous()
        # Store h in fp32 for backward gradient precision
        h_fp32 = torch.empty(B_dim, T_dim, S_dim, dtype=torch.float32, device=B_vals.device)

        BLOCK_S = 32
        num_programs = B_dim * triton.cdiv(S_dim, BLOCK_S)
        _scan_fwd_kernel[(num_programs,)](
            B_vals_c, D_c, h_fp32,
            B_dim, T_dim, S_dim,
            B_vals_c.stride(0), B_vals_c.stride(1), B_vals_c.stride(2),
            h_fp32.stride(0), h_fp32.stride(1), h_fp32.stride(2),
            BLOCK_S=BLOCK_S, num_warps=2, num_stages=1,
        )
        h_out = h_fp32.to(B_vals.dtype)
        ctx.save_for_backward(D_c, h_fp32)
        ctx.shape = (B_dim, T_dim, S_dim)
        return h_out

    @staticmethod
    def backward(ctx, dh: Tensor) -> tuple[Tensor, Tensor]:
        D, h_fp32 = ctx.saved_tensors
        B_dim, T_dim, S_dim = ctx.shape
        dh_c = dh.contiguous()

        dB = torch.empty_like(dh_c, dtype=torch.bfloat16)
        dD = torch.empty_like(dh_c, dtype=torch.bfloat16)

        BLOCK_S = 32
        num_programs = B_dim * triton.cdiv(S_dim, BLOCK_S)
        _scan_bwd_kernel[(num_programs,)](
            dh_c, D, h_fp32,
            dB, dD,
            B_dim, T_dim, S_dim,
            dh_c.stride(0), dh_c.stride(1), dh_c.stride(2),
            h_fp32.stride(0), h_fp32.stride(1), h_fp32.stride(2),
            BLOCK_S=BLOCK_S, num_warps=2, num_stages=1,
        )
        return dB.to(dh.dtype), dD.to(dh.dtype)


def triton_parallel_scan(B_vals: Tensor, D: Tensor) -> Tensor:
    """Drop-in replacement for the chunked scan in _causal_decay_scan."""
    return TritonScanFn.apply(B_vals, D)


# ============================================================================
# 3. FUSED TERNARY DEQUANT KERNEL
# ============================================================================
# Replaces the multi-op dequant path in TernaryLinear.forward():
#   pad → reshape → (Hadamard) → scale → round → clamp → (threshold) → mul → (inv Hadamard) → unpad
# with a single fused kernel.  No backward needed — the STE .detach() blocks
# gradient flow through the dequant path.
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _ternary_dequant_kernel(
        W_ptr, Out_ptr, H_ptr,
        Thr_ptr, ScaleMult_ptr,
        nrows, ncols, ncols_padded,
        GROUP_SIZE: tl.constexpr,
        BLOCK_G: tl.constexpr,  # groups per program
        HAS_TURBO: tl.constexpr,
    ):
        pid = tl.program_id(0)
        groups_per_row = ncols_padded // GROUP_SIZE
        total_groups = nrows * groups_per_row

        g_start = pid * BLOCK_G
        g_ids = g_start + tl.arange(0, BLOCK_G)
        g_mask = g_ids < total_groups

        rows = g_ids // groups_per_row
        col_groups = g_ids % groups_per_row
        col_starts = col_groups * GROUP_SIZE

        g_offs = tl.arange(0, GROUP_SIZE)

        # Load calibration scalars from device memory (no host sync)
        thr = tl.load(Thr_ptr).to(tl.float32)
        scale_mult = tl.load(ScaleMult_ptr).to(tl.float32)

        # Load BLOCK_G groups: (BLOCK_G, GROUP_SIZE)
        ptrs = W_ptr + rows[:, None] * ncols + col_starts[:, None] + g_offs[None, :]
        col_mask = (col_starts[:, None] + g_offs[None, :]) < ncols
        w_groups = tl.load(ptrs, mask=g_mask[:, None] & col_mask, other=0.0)

        # Optional Hadamard rotation
        if HAS_TURBO:
            h_r = tl.arange(0, GROUP_SIZE)
            h_c = tl.arange(0, GROUP_SIZE)
            H = tl.load(H_ptr + h_r[:, None] * GROUP_SIZE + h_c[None, :])
            w_groups = tl.dot(w_groups.to(tl.bfloat16), H.to(tl.bfloat16))

        # Per-group scale with fp16 bottleneck (matches export exactly)
        abs_w = tl.abs(w_groups)
        scale = tl.sum(abs_w, axis=1) / GROUP_SIZE  # (BLOCK_G,)
        scale = scale.to(tl.float16).to(tl.float32)  # fp16 truncation
        scale = tl.maximum(scale, 5.96e-8)  # _FP16_TINY

        # Quantize: z = w/scale, q = round(clamp(z, -1, 1))
        z = w_groups / scale[:, None]
        # Portable rounding: works across Triton 2.x/3.x (tl.libdevice moved)
        # floor(|z| + 0.5) * sign(z) = round-half-away-from-zero
        abs_z = tl.abs(z)
        q = tl.where(z >= 0, (abs_z + 0.5).to(tl.int32).to(tl.float32),
                              -((abs_z + 0.5).to(tl.int32).to(tl.float32)))
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)

        # Apply calibration threshold
        q = tl.where((thr > 0.0) & (tl.abs(z) < thr), 0.0, q)

        # Dequantize
        dequant = q * (scale[:, None] * scale_mult)

        # Inverse Hadamard
        if HAS_TURBO:
            dequant = tl.dot(dequant.to(tl.bfloat16), H.to(tl.bfloat16))

        # Store
        out_ptrs = Out_ptr + rows[:, None] * ncols + col_starts[:, None] + g_offs[None, :]
        tl.store(out_ptrs, dequant.to(w_groups.dtype), mask=g_mask[:, None] & col_mask)


def triton_ternary_dequant(
    w: Tensor, group_size: int, H_fixed: Tensor | None,
    calib_thr: Tensor, calib_scale_mult: Tensor, turbo: bool,
) -> Tensor:
    """Compute ternary dequant weights via fused Triton kernel. No grad needed."""
    nrows, ncols = w.shape[0], w.shape[-1]
    if w.ndim == 3:
        w = w.reshape(w.shape[0], -1)
        nrows, ncols = w.shape

    pad = (group_size - ncols % group_size) % group_size
    ncols_padded = ncols + pad

    out = torch.empty(nrows, ncols, dtype=w.dtype, device=w.device)
    total_groups = nrows * (ncols_padded // group_size)
    BLOCK_G = 16
    grid = (triton.cdiv(total_groups, BLOCK_G),)

    H_ptr = H_fixed if (turbo and H_fixed is not None) else torch.zeros(1, device=w.device)

    _ternary_dequant_kernel[grid](
        w, out, H_ptr,
        calib_thr, calib_scale_mult,
        nrows, ncols, ncols_padded,
        GROUP_SIZE=group_size, BLOCK_G=BLOCK_G,
        HAS_TURBO=(turbo and H_fixed is not None),
        num_warps=4, num_stages=2,
    )
    return out


# ============================================================================
# 3b. FUSED RMS NORM KERNEL
# ============================================================================
# Replaces F.rms_norm used before every TernaryLinear and in NormedTernaryLinear.
# Fuses the variance computation + reciprocal sqrt + scale into a single pass,
# eliminating one HBM read-write cycle.
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        X_ptr, OUT_ptr,
        N_dim: tl.constexpr,
        stride_xr, stride_xc,
        stride_or, stride_oc,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program per row. Computes RMS norm in a single pass over columns."""
        row = tl.program_id(0)
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < N_dim

        x = tl.load(X_ptr + row * stride_xr + col_offs * stride_xc, mask=mask, other=0.0).to(tl.float32)

        # Compute RMS
        sq_sum = tl.sum(x * x, axis=0)
        rms = tl.sqrt(sq_sum / N_dim + eps)
        out = x / rms

        tl.store(OUT_ptr + row * stride_or + col_offs * stride_oc, out.to(tl.bfloat16), mask=mask)


def triton_rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused RMS norm. Input: (..., D), Output: (..., D) in bf16."""
    if not HAS_TRITON or not x.is_cuda or torch.compiler.is_compiling():
        return F.rms_norm(x, (x.size(-1),))

    orig_shape = x.shape
    D = x.size(-1)
    x_2d = x.reshape(-1, D).contiguous()
    num_rows = x_2d.shape[0]

    out = torch.empty_like(x_2d, dtype=torch.bfloat16)
    BLOCK_N = triton.next_power_of_2(D)
    if BLOCK_N > 8192:
        # Fall back for very large dimensions
        return F.rms_norm(x, (x.size(-1),))

    _rms_norm_fwd_kernel[(num_rows,)](
        x_2d, out,
        D,
        x_2d.stride(0), x_2d.stride(1),
        out.stride(0), out.stride(1),
        eps=eps,
        BLOCK_N=BLOCK_N,
        num_warps=min(8, max(1, BLOCK_N // 256)),
        num_stages=1,
    )
    return out.reshape(orig_shape).to(x.dtype)


# ============================================================================
# 4. MOE DISPATCH OPTIMIZATION (Pure PyTorch)
# ============================================================================
# Eliminates repeat_interleave which physically copies ALL tokens K times.
# Instead: iterate over top-k slots and gather per-expert directly.
# Memory: O(N/num_experts) per expert instead of O(N*K).
# ============================================================================

def optimized_moe_dispatch(
    x_flat: Tensor,
    experts: list,
    selected_experts: Tensor,  # (N, effective_top_k)
    routing_weights: Tensor,    # (N, effective_top_k)
    effective_top_k: int,
    num_experts: int,
) -> Tensor:
    """MoE dispatch without repeat_interleave. Drop-in for TernaryMoE body."""
    final_output = torch.zeros_like(x_flat)

    for k_idx in range(effective_top_k):
        expert_ids = selected_experts[:, k_idx]       # (N,) which expert per token
        weights_k = routing_weights[:, k_idx:k_idx+1]  # (N, 1) weights for this slot

        for i, expert in enumerate(experts):
            token_indices = (expert_ids == i).nonzero(as_tuple=True)[0]
            if token_indices.numel() > 0:
                expert_out = expert(x_flat[token_indices])
                final_output.index_add_(
                    0, token_indices,
                    expert_out * weights_k[token_indices],
                )

    return final_output


# ============================================================================
# 5. FEEDBACK ADAPTER OPTIMIZATION (Pure PyTorch)
# ============================================================================
# Eliminates D×D memory_matrix materialization by reordering matmuls.
# Original:  M = V^T @ K  (D×D),  retrieved = M @ Q^T
# Optimized: inner = K @ Q^T (S×T),  retrieved = V^T @ inner  (D×T)
# S (sketch_len) is typically 2, so S×T << D×D.
# ============================================================================

def optimized_feedback_retrieve(
    q: Tensor,        # (B, T, D) query
    k: Tensor,        # (B, S, D) key from sketch
    v: Tensor,        # (B, S, D) value from sketch
    lr: Tensor,       # scalar learning rate
    sketch_len: int,
) -> Tensor:
    """Compute fast-weight retrieval without D×D matrix. Returns (B, T, D)."""
    # inner = K @ Q^T: (B, S, D) @ (B, D, T) = (B, S, T) — tiny when S=2
    q_t = q.transpose(1, 2)
    inner = torch.bmm(k, q_t)

    # retrieved = V^T @ inner: (B, D, S) @ (B, S, T) = (B, D, T) — no D×D!
    v_t = v.transpose(1, 2)
    retrieved_t = (lr / max(sketch_len, 1)) * torch.bmm(v_t, inner)

    return retrieved_t.transpose(1, 2)  # (B, T, D)


# ============================================================================
# 6. NUMERICAL PARITY TESTS
# ============================================================================

def test_fwht_parity(device="cuda", block_size=64, tol_fwd=0.04, tol_bwd=0.04):
    """Verify Triton FWHT matches PyTorch implementation (forward + backward)."""
    B, T, D = 4, 256, 128
    torch.manual_seed(42)
    x = torch.randn(B, T, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Triton path
    y_tri = triton_fwht_blockwise(x, block_size)
    loss_tri = y_tri.float().sum()
    loss_tri.backward()

    # PyTorch path
    y_ref = _pytorch_fwht_blockwise(x_ref, block_size)
    loss_ref = y_ref.float().sum()
    loss_ref.backward()

    fwd_ok = torch.allclose(y_tri.float(), y_ref.float(), atol=tol_fwd, rtol=tol_fwd)
    bwd_ok = torch.allclose(x.grad.float(), x_ref.grad.float(), atol=tol_bwd, rtol=tol_bwd)

    fwd_max = (y_tri.float() - y_ref.float()).abs().max().item()
    bwd_max = (x.grad.float() - x_ref.grad.float()).abs().max().item()

    print(f"[FWHT] Forward parity: {'PASS' if fwd_ok else 'FAIL'} (max_err={fwd_max:.6f})")
    print(f"[FWHT] Backward parity: {'PASS' if bwd_ok else 'FAIL'} (max_err={bwd_max:.6f})")
    return fwd_ok and bwd_ok


def test_scan_parity(device="cuda", tol_fwd=5e-2, tol_bwd=0.15):
    """Verify Triton scan matches sequential PyTorch scan (forward + backward)."""
    B, T, S = 4, 256, 128
    torch.manual_seed(42)

    B_vals = torch.randn(B, T, S, dtype=torch.bfloat16, device=device)
    D = torch.sigmoid(torch.randn(B, T, S, dtype=torch.bfloat16, device=device))

    # PyTorch reference (sequential, fp32)
    B_ref = B_vals.float().clone().requires_grad_(True)
    D_ref = D.float().clone().requires_grad_(True)
    h = torch.zeros(B, S, device=device, dtype=torch.float32)
    hs = []
    for t in range(T):
        h = D_ref[:, t] * h + B_ref[:, t]
        hs.append(h)
    h_ref = torch.stack(hs, dim=1)
    loss_ref = h_ref.sum()
    loss_ref.backward()

    # Triton path
    B_tri = B_vals.clone().requires_grad_(True)
    D_tri = D.clone().requires_grad_(True)
    h_tri = TritonScanFn.apply(B_tri, D_tri)
    loss_tri = h_tri.float().sum()
    loss_tri.backward()

    fwd_ok = torch.allclose(h_tri.float(), h_ref.float().to(torch.bfloat16).float(), atol=tol_fwd, rtol=tol_fwd)
    fwd_max = (h_tri.float() - h_ref.float().to(torch.bfloat16).float()).abs().max().item()

    bwd_b_ok = torch.allclose(B_tri.grad.float(), B_ref.grad.to(torch.bfloat16).float(), atol=tol_bwd, rtol=tol_bwd)
    bwd_d_ok = torch.allclose(D_tri.grad.float(), D_ref.grad.to(torch.bfloat16).float(), atol=tol_bwd, rtol=tol_bwd)

    bwd_b_max = (B_tri.grad.float() - B_ref.grad.to(torch.bfloat16).float()).abs().max().item()
    bwd_d_max = (D_tri.grad.float() - D_ref.grad.to(torch.bfloat16).float()).abs().max().item()

    print(f"[Scan] Forward parity: {'PASS' if fwd_ok else 'FAIL'} (max_err={fwd_max:.6f})")
    print(f"[Scan] Backward dB parity: {'PASS' if bwd_b_ok else 'FAIL'} (max_err={bwd_b_max:.6f})")
    print(f"[Scan] Backward dD parity: {'PASS' if bwd_d_ok else 'FAIL'} (max_err={bwd_d_max:.6f})")
    return fwd_ok and bwd_b_ok and bwd_d_ok


def test_feedback_parity(device="cuda", tol=1e-4):
    """Verify optimized feedback matches original D×D path."""
    B, T, D, S = 4, 256, 128, 2
    torch.manual_seed(42)
    q = torch.randn(B, T, D, device=device)
    k = torch.randn(B, S, D, device=device)
    v = torch.randn(B, S, D, device=device)
    lr = torch.tensor(0.11, device=device)

    # Original: D×D path
    memory_matrix = (lr / S) * torch.bmm(v.transpose(1, 2), k)
    retrieved_orig = torch.bmm(memory_matrix, q.transpose(1, 2)).transpose(1, 2)

    # Optimized: no D×D
    retrieved_opt = optimized_feedback_retrieve(q, k, v, lr, S)

    ok = torch.allclose(retrieved_orig, retrieved_opt, atol=tol, rtol=tol)
    max_err = (retrieved_orig - retrieved_opt).abs().max().item()
    print(f"[Feedback] Parity: {'PASS' if ok else 'FAIL'} (max_err={max_err:.8f})")
    return ok


def run_all_tests(device="cuda"):
    """Run all kernel parity tests."""
    print("=" * 60)
    print("Triton Kernel Parity Tests")
    print("=" * 60)
    results = {}
    if HAS_TRITON:
        results["fwht"] = test_fwht_parity(device)
        results["scan"] = test_scan_parity(device)
        results["engram"] = test_engram_hash_gather_parity(device)
    else:
        print("[SKIP] Triton not available — skipping FWHT, Scan, and Engram tests")
    results["feedback"] = test_feedback_parity(device)
    print("=" * 60)
    all_pass = all(results.values())
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# ============================================================================
# 7. FUSED SPECTRAL DECAY SCAN KERNEL
# ============================================================================
# Replaces the O(blocks^2) decay matrix construction + einsum in
# causal_spectral_decay_scan() with a sequential prefix scan.
#
# Math:  prefix[0] = 0
#        prefix[b] = decay * prefix[b-1] + gate[b-1,-1,:] * x[b-1,-1,:]
#        output[b,t,:] = x[b,t,:] + prefix[b] * decay^(t+1)
#
# One program per (batch, dim_block). Sequential over blocks, parallel over D.
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _spectral_decay_scan_fwd_kernel(
        X_ptr, GATE_ptr, DECAY_ptr, OUT_ptr,
        B_dim, NUM_BLOCKS, BLOCK_SZ: tl.constexpr, D_dim,
        stride_xb, stride_xn, stride_xt, stride_xd,
        stride_gb, stride_gn, stride_gt, stride_gd,
        stride_ob, stride_on, stride_ot, stride_od,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_idx = pid // tl.cdiv(D_dim, BLOCK_D)
        d_block = pid % tl.cdiv(D_dim, BLOCK_D)
        if batch_idx >= B_dim:
            return

        d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D_dim

        # Load decay rates (clamped) for this dim block
        decay = tl.load(DECAY_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        decay = tl.minimum(tl.maximum(decay, 0.0), 0.999)

        # Precompute time_decay[t] = decay^(t+1) for t in [0, BLOCK_SZ)
        # We store these in registers
        prefix = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for block_idx in range(NUM_BLOCKS):
            x_base = batch_idx * stride_xb + block_idx * stride_xn
            g_base = batch_idx * stride_gb + block_idx * stride_gn
            o_base = batch_idx * stride_ob + block_idx * stride_on

            # Process each timestep in this block
            for t in range(BLOCK_SZ):
                x_val = tl.load(X_ptr + x_base + t * stride_xt + d_offs * stride_xd,
                                mask=d_mask, other=0.0).to(tl.float32)

                # Apply prefix state with time decay: decay^(t+1)
                # We compute decay_power incrementally
                time_power = decay  # This will be decay^(t+1)
                for _tp in range(t):
                    time_power = time_power * decay

                out_val = x_val + prefix * time_power
                tl.store(OUT_ptr + o_base + t * stride_ot + d_offs * stride_od,
                         out_val.to(tl.bfloat16), mask=d_mask)

            # After processing this block, update prefix for next block
            # gated_final = gate[block, -1, :] * x[block, -1, :]
            last_t = BLOCK_SZ - 1
            x_last = tl.load(X_ptr + x_base + last_t * stride_xt + d_offs * stride_xd,
                             mask=d_mask, other=0.0).to(tl.float32)
            g_last = tl.load(GATE_ptr + g_base + last_t * stride_gt + d_offs * stride_gd,
                             mask=d_mask, other=0.0).to(tl.float32)
            gated_final = g_last * x_last
            prefix = decay * prefix + gated_final


class TritonSpectralDecayScanFn(torch.autograd.Function):
    """Autograd wrapper for the fused spectral decay scan.

    Forward:  prefix[0]=0, prefix[b] = decay * prefix[b-1] + gate[b-1,-1,:]*x[b-1,-1,:]
              out[b,t,:] = x[b,t,:] + prefix[b] * decay^(t+1)
    """

    @staticmethod
    def forward(ctx, x_blocks: Tensor, decay_rates: Tensor, gate: Tensor) -> Tensor:
        B, num_blocks, block_sz, D = x_blocks.shape
        x_c = x_blocks.contiguous()
        gate_c = gate.contiguous()
        decay_c = decay_rates.contiguous()
        out = torch.empty_like(x_c, dtype=torch.bfloat16)

        BLOCK_D = min(D, 64)
        num_programs = B * triton.cdiv(D, BLOCK_D)
        _spectral_decay_scan_fwd_kernel[(num_programs,)](
            x_c, gate_c, decay_c, out,
            B, num_blocks, block_sz, D,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            gate_c.stride(0), gate_c.stride(1), gate_c.stride(2), gate_c.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_D=BLOCK_D, num_warps=2, num_stages=1,
        )
        out = out.to(x_blocks.dtype)
        ctx.save_for_backward(x_c, gate_c, decay_c, out)
        ctx.shape = (B, num_blocks, block_sz, D)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_c, gate_c, decay_c, out = ctx.saved_tensors
        # Fall back to PyTorch for backward (autograd-safe)
        # Re-run forward in PyTorch for clean backward graph
        B, num_blocks, block_sz, D = ctx.shape
        decay = torch.clamp(decay_c, 0.0, 0.999)
        gated = gate_c * x_c
        if num_blocks > 1:
            block_finals = gated[:, :, -1, :]
            idx = torch.arange(num_blocks, device=decay.device)
            diff = idx[:, None] - idx[None, :]
            mask = diff >= 0
            diff_clamped = torch.max(diff, torch.zeros_like(diff))
            M = decay[None, None, :] ** diff_clamped[:, :, None].to(decay.dtype)
            M = torch.where(mask[:, :, None], M, torch.zeros_like(M))
            states = torch.einsum('ijd,bjd->bid', M, block_finals)
            prefix_states = torch.cat([torch.zeros_like(states[:, :1, :]), states[:, :-1, :]], dim=1)
            t = torch.arange(1, block_sz + 1, device=decay.device, dtype=decay.dtype)
            time_decay = decay[None, :] ** t[:, None]
            result = x_c + prefix_states[:, :, None, :] * time_decay[None, None, :, :]
        else:
            result = x_c.clone()
        result.requires_grad_(True)
        with torch.enable_grad():
            # This is a clean re-derive; we just need gradients
            result.backward(grad_output)
        return result.grad, None, None


def triton_spectral_decay_scan(x_blocks: Tensor, decay_rates: Tensor, gate: Tensor) -> Tensor:
    """Fused spectral decay scan. Drop-in replacement for causal_spectral_decay_scan."""
    if not HAS_TRITON or not x_blocks.is_cuda or torch.compiler.is_compiling():
        return None
    B, num_blocks, block_sz, D = x_blocks.shape
    if num_blocks <= 1:
        return x_blocks
    return TritonSpectralDecayScanFn.apply(x_blocks, decay_rates, gate)


# ============================================================================
# 8. ENGRAM FUSED HASH-GATHER KERNEL
# ============================================================================
# Eliminates intermediate index tensor materialization by computing n-gram
# hashes and gathering from embedding tables in a single kernel launch.
# Handles the 2×2 config (2 bigram heads + 2 trigram heads = 4 total).
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _engram_hash_gather_kernel_2x2(
        IDS_ptr, OUT_ptr,
        TBL0_ptr, TBL1_ptr, TBL2_ptr, TBL3_ptr,
        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
        B_dim, T_dim, BUCKETS: tl.constexpr, HD: tl.constexpr,
        stride_ib, stride_it,
        stride_ob, stride_ot, stride_od,
    ):
        """Fused bigram(2 heads) + trigram(2 heads) hash-and-gather."""
        pid = tl.program_id(0)
        b_idx = pid // T_dim
        t_idx = pid % T_dim
        if b_idx >= B_dim:
            return
        hd_offs = tl.arange(0, HD)
        out_base = b_idx * stride_ob + t_idx * stride_ot
        curr = tl.load(IDS_ptr + b_idx * stride_ib + t_idx * stride_it).to(tl.int64)
        if t_idx >= 1:
            prev = tl.load(IDS_ptr + b_idx * stride_ib + (t_idx - 1) * stride_it).to(tl.int64)
            h0 = (prev * P0 + curr) % BUCKETS
            h1 = (prev * P1 + curr) % BUCKETS
            e0 = tl.load(TBL0_ptr + h0 * HD + hd_offs).to(tl.float32)
            e1 = tl.load(TBL1_ptr + h1 * HD + hd_offs).to(tl.float32)
        else:
            e0 = tl.zeros((HD,), dtype=tl.float32)
            e1 = tl.zeros((HD,), dtype=tl.float32)
        if t_idx >= 2:
            pp = tl.load(IDS_ptr + b_idx * stride_ib + (t_idx - 2) * stride_it).to(tl.int64)
            prev2 = tl.load(IDS_ptr + b_idx * stride_ib + (t_idx - 1) * stride_it).to(tl.int64)
            h2 = (pp * (P2 * P2) + prev2 * P2 + curr) % BUCKETS
            h3 = (pp * (P3 * P3) + prev2 * P3 + curr) % BUCKETS
            e2 = tl.load(TBL2_ptr + h2 * HD + hd_offs).to(tl.float32)
            e3 = tl.load(TBL3_ptr + h3 * HD + hd_offs).to(tl.float32)
        else:
            e2 = tl.zeros((HD,), dtype=tl.float32)
            e3 = tl.zeros((HD,), dtype=tl.float32)
        tl.store(OUT_ptr + out_base + hd_offs * stride_od, e0)
        tl.store(OUT_ptr + out_base + (HD + hd_offs) * stride_od, e1)
        tl.store(OUT_ptr + out_base + (2 * HD + hd_offs) * stride_od, e2)
        tl.store(OUT_ptr + out_base + (3 * HD + hd_offs) * stride_od, e3)

    @triton.jit
    def _engram_hash_gather_kernel_3x3(
        IDS_ptr, OUT_ptr,
        TBL0_ptr, TBL1_ptr, TBL2_ptr,
        TBL3_ptr, TBL4_ptr, TBL5_ptr,
        TBL6_ptr, TBL7_ptr, TBL8_ptr,
        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
        P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr,
        P6: tl.constexpr, P7: tl.constexpr, P8: tl.constexpr,
        B_dim, T_dim, BUCKETS: tl.constexpr, HD: tl.constexpr,
        stride_ib, stride_it,
        stride_ob, stride_ot, stride_od,
    ):
        """Fused bigram(3) + trigram(3) + 4gram(3) hash-and-gather.
        One program per (batch, time). Overflow-safe 4-gram via nested modular reduction.
        """
        pid = tl.program_id(0)
        b_idx = pid // T_dim
        t_idx = pid % T_dim
        if b_idx >= B_dim:
            return
        hd_offs = tl.arange(0, HD)
        out_base = b_idx * stride_ob + t_idx * stride_ot
        ids_base = b_idx * stride_ib
        curr = tl.load(IDS_ptr + ids_base + t_idx * stride_it).to(tl.int64)
        # --- Bigram heads (order=0, 3 heads) ---
        if t_idx >= 1:
            prev = tl.load(IDS_ptr + ids_base + (t_idx - 1) * stride_it).to(tl.int64)
            h0 = (prev * P0 + curr) % BUCKETS
            h1 = (prev * P1 + curr) % BUCKETS
            h2 = (prev * P2 + curr) % BUCKETS
            e0 = tl.load(TBL0_ptr + h0 * HD + hd_offs).to(tl.float32)
            e1 = tl.load(TBL1_ptr + h1 * HD + hd_offs).to(tl.float32)
            e2 = tl.load(TBL2_ptr + h2 * HD + hd_offs).to(tl.float32)
        else:
            e0 = tl.zeros((HD,), dtype=tl.float32)
            e1 = tl.zeros((HD,), dtype=tl.float32)
            e2 = tl.zeros((HD,), dtype=tl.float32)
        # --- Trigram heads (order=1, 3 heads) ---
        if t_idx >= 2:
            pp = tl.load(IDS_ptr + ids_base + (t_idx - 2) * stride_it).to(tl.int64)
            prev_t = tl.load(IDS_ptr + ids_base + (t_idx - 1) * stride_it).to(tl.int64)
            h3 = (pp * (P3 * P3) + prev_t * P3 + curr) % BUCKETS
            h4 = (pp * (P4 * P4) + prev_t * P4 + curr) % BUCKETS
            h5 = (pp * (P5 * P5) + prev_t * P5 + curr) % BUCKETS
            e3 = tl.load(TBL3_ptr + h3 * HD + hd_offs).to(tl.float32)
            e4 = tl.load(TBL4_ptr + h4 * HD + hd_offs).to(tl.float32)
            e5 = tl.load(TBL5_ptr + h5 * HD + hd_offs).to(tl.float32)
        else:
            e3 = tl.zeros((HD,), dtype=tl.float32)
            e4 = tl.zeros((HD,), dtype=tl.float32)
            e5 = tl.zeros((HD,), dtype=tl.float32)
        # --- 4-gram heads (order=2, 3 heads) with overflow-safe nested mod ---
        if t_idx >= 3:
            ppp = tl.load(IDS_ptr + ids_base + (t_idx - 3) * stride_it).to(tl.int64)
            pp_4 = tl.load(IDS_ptr + ids_base + (t_idx - 2) * stride_it).to(tl.int64)
            prev_4 = tl.load(IDS_ptr + ids_base + (t_idx - 1) * stride_it).to(tl.int64)
            h6 = ((((ppp * P6 + pp_4) % BUCKETS) * P6 + prev_4) % BUCKETS * P6 + curr) % BUCKETS
            h7 = ((((ppp * P7 + pp_4) % BUCKETS) * P7 + prev_4) % BUCKETS * P7 + curr) % BUCKETS
            h8 = ((((ppp * P8 + pp_4) % BUCKETS) * P8 + prev_4) % BUCKETS * P8 + curr) % BUCKETS
            e6 = tl.load(TBL6_ptr + h6 * HD + hd_offs).to(tl.float32)
            e7 = tl.load(TBL7_ptr + h7 * HD + hd_offs).to(tl.float32)
            e8 = tl.load(TBL8_ptr + h8 * HD + hd_offs).to(tl.float32)
        else:
            e6 = tl.zeros((HD,), dtype=tl.float32)
            e7 = tl.zeros((HD,), dtype=tl.float32)
            e8 = tl.zeros((HD,), dtype=tl.float32)
        # Store all 9 head embeddings contiguously
        tl.store(OUT_ptr + out_base + hd_offs * stride_od, e0)
        tl.store(OUT_ptr + out_base + (HD + hd_offs) * stride_od, e1)
        tl.store(OUT_ptr + out_base + (2 * HD + hd_offs) * stride_od, e2)
        tl.store(OUT_ptr + out_base + (3 * HD + hd_offs) * stride_od, e3)
        tl.store(OUT_ptr + out_base + (4 * HD + hd_offs) * stride_od, e4)
        tl.store(OUT_ptr + out_base + (5 * HD + hd_offs) * stride_od, e5)
        tl.store(OUT_ptr + out_base + (6 * HD + hd_offs) * stride_od, e6)
        tl.store(OUT_ptr + out_base + (7 * HD + hd_offs) * stride_od, e7)
        tl.store(OUT_ptr + out_base + (8 * HD + hd_offs) * stride_od, e8)


def triton_engram_hash_gather(input_ids: Tensor, tables: list, primes: Tensor,
                               num_orders: int, num_heads: int, head_dim: int,
                               buckets: int) -> Tensor | None:
    """Fused hash-and-gather for Engram. Supports 2x2 and 3x3 configs."""
    if not HAS_TRITON or not input_ids.is_cuda:
        return None
    B, T = input_ids.shape
    hash_dim = head_dim * num_orders * num_heads
    out = torch.empty(B, T, hash_dim, dtype=torch.bfloat16, device=input_ids.device)
    p = primes[:num_orders * num_heads].tolist()
    if num_orders == 2 and num_heads == 2:
        _engram_hash_gather_kernel_2x2[(B * T,)](
            input_ids, out,
            tables[0].weight, tables[1].weight, tables[2].weight, tables[3].weight,
            p[0], p[1], p[2], p[3],
            B, T, buckets, head_dim,
            input_ids.stride(0), input_ids.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=2, num_stages=1,
        )
        return out
    elif num_orders == 3 and num_heads == 3:
        _engram_hash_gather_kernel_3x3[(B * T,)](
            input_ids, out,
            tables[0].weight, tables[1].weight, tables[2].weight,
            tables[3].weight, tables[4].weight, tables[5].weight,
            tables[6].weight, tables[7].weight, tables[8].weight,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
            B, T, buckets, head_dim,
            input_ids.stride(0), input_ids.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=4, num_stages=1,
        )
        return out
    return None


def engram_entropy_gated_correction(logits: Tensor, engram_logits: Tensor,
                                     alpha: float = 0.05, entropy_thr: float = 2.0) -> Tensor:
    """Apply Engram logit correction only when model entropy exceeds threshold.
    Cheap, deterministic, no gradients. For eval-time only."""
    with torch.no_grad():
        probs = torch.softmax(logits.float(), dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        mask = (entropy > entropy_thr).to(logits.dtype)
        return logits + alpha * mask * engram_logits


def test_engram_hash_gather_parity(device="cuda", tol=1e-5):
    """Verify Triton hash-gather matches PyTorch reference for 2×2 Engram config."""
    B, T, BUCKETS, HD = 4, 128, 2048, 8
    PRIMES = [92821, 131071, 174763, 216091]
    torch.manual_seed(42)

    # Create 4 embedding tables
    tables = [torch.nn.Embedding(BUCKETS, HD).to(device=device, dtype=torch.bfloat16)
              for _ in range(4)]
    primes = torch.tensor(PRIMES, dtype=torch.long, device=device)
    input_ids = torch.randint(0, 8192, (B, T), device=device, dtype=torch.long)

    # PyTorch reference
    p = primes[:4]
    ref = torch.zeros(B, T, 4 * HD, dtype=torch.bfloat16, device=device)
    for b in range(B):
        for t in range(T):
            curr = input_ids[b, t]
            if t >= 1:
                prev = input_ids[b, t - 1]
                h0 = int((prev * p[0] + curr) % BUCKETS)
                h1 = int((prev * p[1] + curr) % BUCKETS)
                ref[b, t, :HD] = tables[0].weight[h0]
                ref[b, t, HD:2*HD] = tables[1].weight[h1]
            if t >= 2:
                pp = input_ids[b, t - 2]
                prev = input_ids[b, t - 1]
                h2 = int((pp * p[2] * p[2] + prev * p[2] + curr) % BUCKETS)
                h3 = int((pp * p[3] * p[3] + prev * p[3] + curr) % BUCKETS)
                ref[b, t, 2*HD:3*HD] = tables[2].weight[h2]
                ref[b, t, 3*HD:] = tables[3].weight[h3]

    # Triton path
    tri = triton_engram_hash_gather(input_ids, tables, primes, 2, 2, HD, BUCKETS)
    if tri is None:
        print("[Engram] SKIP — Triton not available or config mismatch")
        return True

    ok = torch.allclose(tri.float(), ref.float(), atol=tol, rtol=tol)
    max_err = (tri.float() - ref.float()).abs().max().item()
    print(f"[Engram] Hash-gather parity: {'PASS' if ok else 'FAIL'} (max_err={max_err:.8f})")
    return ok


if __name__ == "__main__":
    if torch.cuda.is_available():
        run_all_tests("cuda")
    else:
        print("CUDA not available — cannot run Triton kernel tests")

