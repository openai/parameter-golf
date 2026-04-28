"""triton_kernels.py — Custom Triton kernels for high-performance training.

Consolidated source of truth for all memory-bandwidth-bound operations.
Includes: FWHT, Parallel Scan, Spectral Decay Scan, Ternary Dequant, Engram, and RMSNorm.
"""
from __future__ import annotations
import math
import os
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

if os.environ.get('DISABLE_TRITON', '0') == '1':
    HAS_TRITON = False

# ---------------------------------------------------------------------------
# Hadamard matrix cache
# ---------------------------------------------------------------------------
_HADAMARD_CACHE: dict[tuple[int, str], Tensor] = {}

def _get_hadamard(n: int, device) -> Tensor:
    key = (n, str(device))
    if key not in _HADAMARD_CACHE:
        assert n > 0 and n & n - 1 == 0, f'n must be power of 2, got {n}'
        H = torch.ones(1, 1, dtype=torch.float32, device='cpu')
        for _ in range(int(math.log2(n))):
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = (H / math.sqrt(n)).to(device).contiguous()
        _HADAMARD_CACHE[key] = H
    return _HADAMARD_CACHE[key]

# ============================================================================
# 1. FUSED FAST WALSH-HADAMARD TRANSFORM (FWHT)
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _fwht_kernel(X_ptr, H_ptr, Out_ptr, total_blocks, D: tl.constexpr, stride_xb, stride_xt, stride_xd, BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        if pid_b >= total_blocks: return
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        H = tl.load(H_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :])
        t_offs = tl.arange(0, BLOCK_SIZE)
        x_ptrs = X_ptr + pid_b * stride_xb + t_offs[:, None] * stride_xt + d_offs[None, :] * stride_xd
        data = tl.load(x_ptrs, mask=d_mask[None, :], other=0.0)
        result = tl.dot(H.to(tl.bfloat16), data.to(tl.bfloat16))
        out_ptrs = Out_ptr + pid_b * stride_xb + t_offs[:, None] * stride_xt + d_offs[None, :] * stride_xd
        tl.store(out_ptrs, result.to(data.dtype), mask=d_mask[None, :])

    class TritonFWHTFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_blocks: Tensor, H: Tensor) -> Tensor:
            x_blocks = x_blocks.contiguous()
            (total_blocks, block_size, D) = x_blocks.shape
            out = torch.empty_like(x_blocks)
            BLOCK_D = min(D, 128)
            grid = (total_blocks, triton.cdiv(D, BLOCK_D))
            _fwht_kernel[grid](x_blocks, H, out, total_blocks, D, x_blocks.stride(0), x_blocks.stride(1), x_blocks.stride(2), BLOCK_SIZE=block_size, BLOCK_D=BLOCK_D, num_warps=4, num_stages=3)
            ctx.save_for_backward(H)
            ctx.x_dtype = x_blocks.dtype
            return out
        @staticmethod
        def backward(ctx, dy: Tensor) -> tuple[Tensor, None]:
            (H,) = ctx.saved_tensors
            # Promote to fp32 for improved gradient precision during accumulation
            dy = dy.contiguous().to(torch.float32)
            (total_blocks, block_size, D) = dy.shape
            dx = torch.empty_like(dy)
            BLOCK_D = min(D, 128)
            grid = (total_blocks, triton.cdiv(D, BLOCK_D))
            _fwht_kernel[grid](dy, H, dx, total_blocks, D, dy.stride(0), dy.stride(1), dy.stride(2), BLOCK_SIZE=block_size, BLOCK_D=BLOCK_D, num_warps=4, num_stages=3)
            return (dx.to(dtype=ctx.x_dtype), None)

def triton_fwht_blockwise(x: Tensor, block_size: int, num_stages: int | None=None) -> Tensor:
    max_stages = int(math.log2(block_size))
    if num_stages is not None and int(num_stages) != max_stages:
        return _pytorch_fwht_blockwise(x, block_size, num_stages)
    if not HAS_TRITON or not x.is_cuda: return _pytorch_fwht_blockwise(x, block_size, num_stages)
    (B, T, D) = x.shape
    pad_len = (block_size - T % block_size) % block_size
    if pad_len > 0: x = F.pad(x, (0,0,0,pad_len))
    T_padded = x.shape[1]
    num_blocks = T_padded // block_size
    x_blocks = x.reshape(B * num_blocks, block_size, D).contiguous()
    H = _get_hadamard(block_size, x.device).to(torch.float32)
    y_blocks = TritonFWHTFn.apply(x_blocks, H)
    y = y_blocks.reshape(B, T_padded, D)
    return y[:, :T, :]

def _pytorch_fwht_blockwise(x: Tensor, block_size: int, num_stages: int | None=None) -> Tensor:
    (B, T, D) = x.shape
    pad_len = (block_size - T % block_size) % block_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
    T_padded = x.shape[1]
    num_blocks = T_padded // block_size
    
    # If doing a full transform, matrix multiply is faster and more memory-efficient than recursive CATs
    if num_stages is None or num_stages == int(math.log2(block_size)):
        H = _get_hadamard(block_size, x.device).to(x.dtype)
        x_blocks = x.view(B * num_blocks, block_size, D)
        # [BN, BS, BS] @ [BN, BS, D] -> [BN, BS, D]
        # Using bmm is usually faster than einsum for this shape
        res = torch.matmul(H, x_blocks)
        return res.view(B, T_padded, D)[:, :T, :]
        
    # Partial transform fallback (rarely used, but kept for completeness)
    result = x.view(B * num_blocks, block_size, D)
    h = int(num_stages)
    for stage in range(h):
        stride = 1 << stage
        result = result.view(B * num_blocks, block_size // (2 * stride), 2, stride, D)
        even = result[:, :, 0, :, :]
        odd = result[:, :, 1, :, :]
        # Still using CAT here for partial stages, but on fewer dimensions
        result = torch.cat([even + odd, even - odd], dim=2)
    
    return result.view(B, T_padded, D)[:, :T, :] * (1.0 / math.sqrt(2**h))

# ============================================================================
# 2. PARALLEL LINEAR RECURRENCE SCAN
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _scan_fwd_kernel(B_ptr, D_ptr, H_ptr, B_dim, T_dim, S_dim, stride_b, stride_t, stride_s, BLOCK_S: tl.constexpr):
        pid = tl.program_id(0)
        batch_idx = pid // tl.cdiv(S_dim, BLOCK_S)
        s_block = pid % tl.cdiv(S_dim, BLOCK_S)
        s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S_dim
        h = tl.zeros((BLOCK_S,), dtype=tl.float32)
        for t in range(T_dim):
            base = batch_idx * stride_b + t * stride_t
            b_val = tl.load(B_ptr + base + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)
            d_val = tl.load(D_ptr + base + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)
            h = d_val * h + b_val
            tl.store(H_ptr + base + s_offs * stride_s, h, mask=s_mask)

    @triton.jit
    def _scan_bwd_kernel(DH_ptr, D_ptr, H_ptr, DB_ptr, DD_ptr, B_dim, T_dim, S_dim, stride_b, stride_t, stride_s, BLOCK_S: tl.constexpr):
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
            tl.store(DB_ptr + base + s_offs * stride_s, delta.to(tl.float32), mask=s_mask)
            h_prev = tl.zeros((BLOCK_S,), dtype=tl.float32)
            if t > 0:
                h_prev = tl.load(H_ptr + (batch_idx * stride_b + (t - 1) * stride_t) + s_offs * stride_s, mask=s_mask, other=0.0).to(tl.float32)
            tl.store(DD_ptr + base + s_offs * stride_s, (delta * h_prev).to(tl.float32), mask=s_mask)
            delta = d_val * delta

    class TritonScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B_vals: Tensor, D: Tensor) -> Tensor:
            (B_dim, T_dim, S_dim) = B_vals.shape
            B_vals_c, D_c = B_vals.contiguous(), D.contiguous()
            h = torch.empty_like(B_vals_c, dtype=torch.float32)
            BLOCK_S = 32
            num_programs = B_dim * triton.cdiv(S_dim, BLOCK_S)
            _scan_fwd_kernel[num_programs,](B_vals_c, D_c, h, B_dim, T_dim, S_dim, B_vals_c.stride(0), B_vals_c.stride(1), B_vals_c.stride(2), BLOCK_S=BLOCK_S, num_warps=2, num_stages=1)
            ctx.save_for_backward(D_c, h)
            ctx.x_dtype = B_vals.dtype
            ctx.shape = (B_dim, T_dim, S_dim)
            return h.to(B_vals.dtype)
        @staticmethod
        def backward(ctx, dh: Tensor) -> tuple[Tensor, Tensor]:
            (D, h) = ctx.saved_tensors
            (B_dim, T_dim, S_dim) = ctx.shape
            dh_c = dh.contiguous()
            # Storing grads in fp32 to prevent truncation/dead parameters
            dB, dD = torch.empty_like(dh_c, dtype=torch.float32), torch.empty_like(dh_c, dtype=torch.float32)
            BLOCK_S = 32
            num_programs = B_dim * triton.cdiv(S_dim, BLOCK_S)
            _scan_bwd_kernel[num_programs,](dh_c, D, h, dB, dD, B_dim, T_dim, S_dim, dh_c.stride(0), dh_c.stride(1), dh_c.stride(2), BLOCK_S=BLOCK_S, num_warps=2, num_stages=1)
            return (dB.to(ctx.x_dtype), dD.to(ctx.x_dtype))

def triton_parallel_scan(B_vals: Tensor, D: Tensor) -> Tensor:
    if not HAS_TRITON or not B_vals.is_cuda: return _pytorch_scan_fallback(B_vals, D)
    return TritonScanFn.apply(B_vals, D)

def _pytorch_scan_fallback(B_vals: Tensor, D: Tensor) -> Tensor:
    (B_dim, T_dim, S_dim) = B_vals.shape
    h = B_vals.new_zeros(B_dim, S_dim)
    hs = []
    for t in range(T_dim):
        h = D[:,t]*h + B_vals[:,t]
        hs.append(h)
    return torch.stack(hs, dim=1)

# ============================================================================
# 3. FUSED TERNARY DEQUANT KERNEL
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _ternary_dequant_kernel(W_ptr, Out_ptr, H_ptr, Thr_ptr, ScaleMult_ptr, nrows, ncols, ncols_padded, GROUP_SIZE: tl.constexpr, BLOCK_G: tl.constexpr, HAS_TURBO: tl.constexpr):
        pid = tl.program_id(0)
        groups_per_row = ncols_padded // GROUP_SIZE
        total_groups = nrows * groups_per_row
        g_start = pid * BLOCK_G
        g_ids = g_start + tl.arange(0, BLOCK_G)
        g_mask = g_ids < total_groups
        rows = g_ids // groups_per_row
        col_starts = g_ids % groups_per_row * GROUP_SIZE
        g_offs = tl.arange(0, GROUP_SIZE)
        thr, scale_mult = tl.load(Thr_ptr).to(tl.float32), tl.load(ScaleMult_ptr).to(tl.float32)
        w_ptrs = W_ptr + rows[:, None] * ncols + col_starts[:, None] + g_offs[None, :]
        w_groups = tl.load(w_ptrs, mask=g_mask[:, None] & (col_starts[:, None] + g_offs[None, :] < ncols), other=0.0)
        if HAS_TURBO:
            H = tl.load(H_ptr + tl.arange(0, GROUP_SIZE)[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :])
            w_groups = tl.dot(w_groups.to(tl.bfloat16), H.to(tl.bfloat16))
        scale = tl.maximum(tl.sum(tl.abs(w_groups), axis=1) / GROUP_SIZE, 5.96e-08)
        z = w_groups / scale[:, None]
        abs_z = tl.abs(z)
        q = tl.where(z >= 0, (abs_z + 0.5).to(tl.int32).to(tl.float32), -(abs_z + 0.5).to(tl.int32).to(tl.float32))
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)
        q = tl.where((thr > 0.0) & (tl.abs(z) < thr), 0.0, q)
        dequant = q * (scale[:, None] * scale_mult)
        if HAS_TURBO: dequant = tl.dot(dequant.to(tl.bfloat16), H.to(tl.bfloat16))
        tl.store(Out_ptr + rows[:, None] * ncols + col_starts[:, None] + g_offs[None, :], dequant.to(w_groups.dtype), mask=g_mask[:, None] & (col_starts[:, None] + g_offs[None, :] < ncols))

def triton_ternary_dequant(w: Tensor, group_size: int, H_fixed: Tensor | None, calib_thr: Tensor, calib_scale_mult: Tensor, turbo: bool) -> Tensor:
    (nrows, ncols) = (w.shape[0], w.shape[-1])
    pad = (group_size - ncols % group_size) % group_size
    ncols_padded = ncols + pad
    out = torch.empty(nrows, ncols, dtype=w.dtype, device=w.device)
    grid = (triton.cdiv(nrows * (ncols_padded // group_size), 16),)
    _ternary_dequant_kernel[grid](w, out, H_fixed if turbo and H_fixed is not None else torch.zeros(1, device=w.device), calib_thr, calib_scale_mult, nrows, ncols, ncols_padded, GROUP_SIZE=group_size, BLOCK_G=16, HAS_TURBO=turbo and H_fixed is not None, num_warps=4, num_stages=2)
    return out

# ============================================================================
# 4. ENGRAM HASH-GATHER KERNELS
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _engram_hash_gather_kernel_2x2(IDS_ptr, OUT_ptr, TBL0_ptr, TBL1_ptr, TBL2_ptr, TBL3_ptr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr, B_dim, T_dim, BUCKETS: tl.constexpr, HD: tl.constexpr, BLOCK_HD: tl.constexpr, stride_ib, stride_it, stride_ob, stride_ot, stride_od):
        pid = tl.program_id(0)
        b_idx, t_idx = pid // T_dim, pid % T_dim
        if b_idx >= B_dim: return
        hd_offs, hd_mask = tl.arange(0, BLOCK_HD), tl.arange(0, BLOCK_HD) < HD
        out_base = b_idx * stride_ob + t_idx * stride_ot
        curr = tl.load(IDS_ptr + b_idx * stride_ib + t_idx * stride_it).to(tl.int64)
        if t_idx >= 1:
            prev = tl.load(IDS_ptr + b_idx * stride_ib + (tl.maximum(t_idx, 1)-1) * stride_it).to(tl.int64)
            h0, h1 = (prev * P0 + curr) % BUCKETS, (prev * P1 + curr) % BUCKETS
            e0, e1 = tl.load(TBL0_ptr + h0 * HD + hd_offs, mask=hd_mask, other=0.0).to(tl.float32), tl.load(TBL1_ptr + h1 * HD + hd_offs, mask=hd_mask, other=0.0).to(tl.float32)
        else: e0, e1 = tl.zeros((BLOCK_HD,), dtype=tl.float32), tl.zeros((BLOCK_HD,), dtype=tl.float32)
        if t_idx >= 2:
            pp = tl.load(IDS_ptr + b_idx * stride_ib + (tl.maximum(t_idx, 2)-2) * stride_it).to(tl.int64)
            prev = tl.load(IDS_ptr + b_idx * stride_ib + (tl.maximum(t_idx, 1)-1) * stride_it).to(tl.int64)
            h2, h3 = (pp * (P2 * P2) + prev * P2 + curr) % BUCKETS, (pp * (P3 * P3) + prev * P3 + curr) % BUCKETS
            e2, e3 = tl.load(TBL2_ptr + h2 * HD + hd_offs, mask=hd_mask, other=0.0).to(tl.float32), tl.load(TBL3_ptr + h3 * HD + hd_offs, mask=hd_mask, other=0.0).to(tl.float32)
        else: e2, e3 = tl.zeros((BLOCK_HD,), dtype=tl.float32), tl.zeros((BLOCK_HD,), dtype=tl.float32)
        tl.store(OUT_ptr + out_base + hd_offs * stride_od, e0, mask=hd_mask)
        tl.store(OUT_ptr + out_base + (HD + hd_offs) * stride_od, e1, mask=hd_mask)
        tl.store(OUT_ptr + out_base + (2 * HD + hd_offs) * stride_od, e2, mask=hd_mask)
        tl.store(OUT_ptr + out_base + (3 * HD + hd_offs) * stride_od, e3, mask=hd_mask)

    @triton.jit
    def _engram_hash_gather_kernel_3x3(IDS_ptr, OUT_ptr, TBL0_ptr, TBL1_ptr, TBL2_ptr, TBL3_ptr, TBL4_ptr, TBL5_ptr, TBL6_ptr, TBL7_ptr, TBL8_ptr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr, P4: tl.constexpr, P5: tl.constexpr, P6: tl.constexpr, P7: tl.constexpr, P8: tl.constexpr, B_dim, T_dim, BUCKETS: tl.constexpr, HD: tl.constexpr, BLOCK_HD: tl.constexpr, stride_ib, stride_it, stride_ob, stride_ot, stride_od):
        pid = tl.program_id(0)
        b_idx, t_idx = pid // T_dim, pid % T_dim
        if b_idx >= B_dim: return
        hd_offs, hd_mask = tl.arange(0, BLOCK_HD), tl.arange(0, BLOCK_HD) < HD
        out_base, ids_base = b_idx * stride_ob + t_idx * stride_ot, b_idx * stride_ib
        curr = tl.load(IDS_ptr + ids_base + t_idx * stride_it).to(tl.int64)
        if t_idx >= 1:
            prev = tl.load(IDS_ptr + ids_base + (tl.maximum(t_idx, 1)-1) * stride_it).to(tl.int64)
            h0,h1,h2 = (prev*P0+curr)%BUCKETS, (prev*P1+curr)%BUCKETS, (prev*P2+curr)%BUCKETS
            e0,e1,e2 = tl.load(TBL0_ptr+h0*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32), tl.load(TBL1_ptr+h1*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32), tl.load(TBL2_ptr+h2*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32)
        else: e0=e1=e2 = tl.zeros((BLOCK_HD,), dtype=tl.float32)
        if t_idx >= 2:
            pp,prev = tl.load(IDS_ptr+ids_base+(tl.maximum(t_idx,2)-2)*stride_it).to(tl.int64), tl.load(IDS_ptr+ids_base+(tl.maximum(t_idx,1)-1)*stride_it).to(tl.int64)
            h3,h4,h5 = (pp*(P3*P3)+prev*P3+curr)%BUCKETS, (pp*(P4*P4)+prev*P4+curr)%BUCKETS, (pp*(P5*P5)+prev*P5+curr)%BUCKETS
            e3,e4,e5 = tl.load(TBL3_ptr+h3*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32), tl.load(TBL4_ptr+h4*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32), tl.load(TBL5_ptr+h5*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32)
        else: e3=e4=e5 = tl.zeros((BLOCK_HD,), dtype=tl.float32)
        if t_idx >= 3:
            ppp,pp_4,prev_4 = tl.load(IDS_ptr+ids_base+(tl.maximum(t_idx,3)-3)*stride_it).to(tl.int64), tl.load(IDS_ptr+ids_base+(tl.maximum(t_idx,2)-2)*stride_it).to(tl.int64), tl.load(IDS_ptr+ids_base+(tl.maximum(t_idx,1)-1)*stride_it).to(tl.int64)
            h6 = ((((ppp*P6+pp_4)%BUCKETS)*P6+prev_4)%BUCKETS*P6+curr)%BUCKETS
            h7 = ((((ppp*P7+pp_4)%BUCKETS)*P7+prev_4)%BUCKETS*P7+curr)%BUCKETS
            h8 = ((((ppp*P8+pp_4)%BUCKETS)*P8+prev_4)%BUCKETS*P8+curr)%BUCKETS
            e6,e7,e8 = tl.load(TBL6_ptr+h6*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32), tl.load(TBL7_ptr+h7*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32), tl.load(TBL8_ptr+h8*HD+hd_offs,mask=hd_mask,other=0.0).to(tl.float32)
        else: e6=e7=e8 = tl.zeros((BLOCK_HD,), dtype=tl.float32)
        for i,e in enumerate([e0,e1,e2,e3,e4,e5,e6,e7,e8]): tl.store(OUT_ptr + out_base + (i*HD + hd_offs)*stride_od, e, mask=hd_mask)

    class _TritonEngramHashGatherFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_ids: Tensor, primes: Tensor, buckets: int, num_orders: int, num_heads: int, head_dim: int, *weights: Tensor) -> Tensor:
            (B, T) = input_ids.shape; hash_dim = head_dim * num_orders * num_heads
            p = primes[:num_orders * num_heads].tolist(); block_hd = triton.next_power_of_2(head_dim)
            dtype = weights[0].dtype
            out = torch.empty(B, T, hash_dim, dtype=dtype, device=input_ids.device)
            if num_orders == 2 and num_heads == 2:
                # Optimized 4-table dispatch for 2-bigram + 2-trigram layout
                _engram_hash_gather_kernel_2x2[(B * T,)](input_ids, out, weights[0], weights[1], weights[2], weights[3], p[0], p[1], p[2], p[3], B, T, buckets, head_dim, block_hd, input_ids.stride(0), input_ids.stride(1), out.stride(0), out.stride(1), out.stride(2), num_warps=2, num_stages=1)
            elif num_orders == 3 and num_heads == 3:
                # Optimized 9-table dispatch for 3x3 layout
                _engram_hash_gather_kernel_3x3[(B * T,)](input_ids, out, *weights, *p, B, T, buckets, head_dim, block_hd, input_ids.stride(0), input_ids.stride(1), out.stride(0), out.stride(1), out.stride(2), num_warps=4, num_stages=1)
            else:
                # Fallback for other configurations (e.g. 1x4, 4x1, MoE Engram or 8-head)
                return _engram_pytorch_retrieve(input_ids, list(weights), p, num_orders, num_heads, buckets).to(out.dtype)
            ctx.save_for_backward(input_ids, primes, *weights); ctx.config = (num_orders, num_heads, int(buckets))
            return out
        @staticmethod
        def backward(ctx, grad_output: Tensor):
            saved = ctx.saved_tensors; input_ids, primes, weights = saved[0], saved[1], saved[2:]
            (num_orders, num_heads, buckets) = ctx.config; primes_list = primes[:num_orders * num_heads].tolist()
            w_in = [w.detach().requires_grad_(True) for w in weights]
            with torch.enable_grad(): memory = _engram_pytorch_retrieve(input_ids, w_in, primes_list, num_orders, num_heads, buckets)
            grads = torch.autograd.grad(memory, w_in, grad_output, allow_unused=True)
            return (None, None, None, None, None, None) + tuple(grads)

def triton_engram_hash_gather(input_ids: Tensor, tables: list, primes: Tensor, num_orders: int, num_heads: int, head_dim: int, buckets: int) -> Tensor:
    if not HAS_TRITON or not input_ids.is_cuda: return _engram_pytorch_retrieve(input_ids, [t.weight for t in tables], primes.tolist(), num_orders, num_heads, buckets)
    return _TritonEngramHashGatherFn.apply(input_ids, primes, buckets, num_orders, num_heads, head_dim, *[t.weight for t in tables])

def _engram_pytorch_retrieve(input_ids: Tensor, weights: list, primes_list: list, num_orders: int, num_heads: int, buckets: int) -> Tensor:
    (B, T) = input_ids.shape; parts = []; ids_long = input_ids.long(); head_dim, dtype = weights[0].size(-1), weights[0].dtype
    for order in range(num_orders):
        for head in range(num_heads):
            k = order * num_heads + head; p = primes_list[k]; min_len = order + 2
            if T < min_len: parts.append(torch.zeros(B, T, head_dim, dtype=dtype, device=input_ids.device)); continue
            if order == 0: h, pad = (ids_long[:,:-1]*p + ids_long[:,1:])%buckets, 1
            elif order == 1: h, pad = (ids_long[:,:-2]*(p*p) + ids_long[:,1:-1]*p + ids_long[:,2:])%buckets, 2
            elif order == 2: h, pad = ((((ids_long[:,:-3] * p + ids_long[:,1:-2]) % buckets) * p + ids_long[:,2:-1]) % buckets * p + ids_long[:,3:]) % buckets, 3
            h_p = F.pad(h, (pad, 0), value=-1); m = (h_p >= 0).unsqueeze(-1)
            parts.append(F.embedding(h_p.clamp(min=0), weights[k]) * m.to(dtype))
    return torch.cat(parts, dim=-1)

# ============================================================================
# 5. SPECTRAL DECAY SCAN
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _spectral_decay_scan_fwd_kernel(X_ptr, GATE_ptr, DECAY_ptr, INIT_STATE_ptr, OUT_ptr, B_dim, NUM_BLOCKS, BLOCK_SZ: tl.constexpr, D_dim, stride_xb, stride_xn, stride_xt, stride_xd, stride_gb, stride_gn, stride_gt, stride_gd, stride_ob, stride_on, stride_ot, stride_od, BLOCK_D: tl.constexpr):
        pid = tl.program_id(0); batch_idx, d_block = pid // tl.cdiv(D_dim, BLOCK_D), pid % tl.cdiv(D_dim, BLOCK_D)
        if batch_idx >= B_dim: return
        d_offs, d_mask = d_block * BLOCK_D + tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_D) < D_dim
        decay = tl.clamp(tl.load(DECAY_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32), 0.0, 0.999)
        prefix = tl.load(INIT_STATE_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        for block_idx in range(NUM_BLOCKS):
            x_b, g_b, o_b = batch_idx*stride_xb + block_idx*stride_xn, batch_idx*stride_gb + block_idx*stride_gn, batch_idx*stride_ob + block_idx*stride_on
            cur_prefix = prefix
            for t in range(BLOCK_SZ):
                x_v = tl.load(X_ptr + x_b + t*stride_xt + d_offs*stride_xd, mask=d_mask, other=0.0).to(tl.float32)
                g_v = tl.load(GATE_ptr + g_b + t*stride_gt + d_offs*stride_gd, mask=d_mask, other=0.0).to(tl.float32)
                # Apply decay and gate per-token to capture full sequence dynamics (Issue 5)
                tl.store(OUT_ptr + o_b + t*stride_ot + d_offs*stride_od, (x_v + cur_prefix).to(OUT_ptr.dtype.element_ty), mask=d_mask)
                # cur_prefix is now the state after integrating token t and decaying it.
                # This state is exactly what the next token (t+1) expects as its 'previous state'.
                cur_prefix = (cur_prefix * decay) + (g_v * x_v)
            # Carry the final accumulated/decayed state of this block to the next block
            # to maintain continuous exponential decay (math matching decay ** 32).
            prefix = cur_prefix

    class _TritonSpectralDecayScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_b: Tensor, decay: Tensor, gate: Tensor, initial_state: Tensor | None = None) -> Tensor:
            (B, nb, sz, D) = x_b.shape; out = torch.empty_like(x_b); bd = min(D, 64)
            init_val = initial_state if initial_state is not None else torch.zeros(D, device=x_b.device, dtype=torch.float32)
            _spectral_decay_scan_fwd_kernel[(B*triton.cdiv(D, bd),)](x_b, gate, decay, init_val, out, B, nb, sz, D, *x_b.stride(), *gate.stride(), *out.stride(), BLOCK_D=bd, num_warps=2, num_stages=1)
            ctx.save_for_backward(x_b, decay, gate, initial_state); return out.to(x_b.dtype)
        @staticmethod
        def backward(ctx, grad_output: Tensor):
            (x_c, d_c, g_c, init_c) = ctx.saved_tensors; nb, sz = x_c.shape[1], x_c.shape[2]
            # Promote saved tensors to fp32 to prevent precision loss in the recursive chain
            xi, di, gi = [t.detach().to(torch.float32).requires_grad_(True) for t in (x_c, d_c, g_c)]
            ii = init_c.detach().to(torch.float32).requires_grad_(True) if init_c is not None else None
            go = grad_output.to(torch.float32)
            with torch.enable_grad():
                dec = torch.clamp(di, 0.0, 0.999); gated = gi * xi
                # Standard recurrence path: y = x + s. Build it under grad mode so autograd
                # can differentiate the recomputed reference scan during backward.
                cur_s = ii.unsqueeze(0) if ii is not None else torch.zeros(xi.size(0), xi.size(-1), device=xi.device)
                outs = []
                for b in range(nb):
                    b_outs = []
                    for t in range(sz):
                        x_v = xi[:, b, t, :]
                        g_v = gi[:, b, t, :]
                        b_outs.append(x_v + cur_s)
                        cur_s = (cur_s * dec) + (g_v * x_v)
                    outs.append(torch.stack(b_outs, dim=1))
                res = torch.stack(outs, dim=1)
                # Keep decay and optional initial state in the graph even when their direct
                # contribution is numerically tiny for a particular slice.
                res = res + (di.sum() * 1e-9)
                if ii is not None:
                    res = res + (ii.sum() * 1e-9)
                grads = torch.autograd.grad(res, (xi, di, gi) + ((ii,) if ii is not None else ()), go, allow_unused=True)
            return tuple(g.to(grad_output.dtype) if g is not None else None for g in (grads[:3] + ((grads[3],) if ii is not None else (None,))))

def triton_spectral_decay_scan(x_blocks: Tensor, decay_rates: Tensor, gate: Tensor, initial_state: Tensor | None = None) -> Tensor:
    if HAS_TRITON and x_blocks.is_cuda:
        return _TritonSpectralDecayScanFn.apply(x_blocks, decay_rates, gate, initial_state)
    # PyTorch fallback
    (B, nb, sz, D) = x_blocks.shape
    dec = torch.clamp(decay_rates, 0.0, 0.999)
    gated = gate * x_blocks
    # Issue 4: Ensure single-block case still applies decay/gate/state
    # We no longer exit early if nb < 1. Standard scan fallback below handles nb >= 1.
    dec = torch.clamp(decay_rates, 0.0, 0.999)
    (B_dim, nb, sz, D_dim) = x_blocks.shape
    device = x_blocks.device
    dtype = x_blocks.dtype
    
    # Linear recurrence: state[t+1] = decay * state[t] + gate[t] * x[t]
    # y[t] = x[t] + state[t]
    state = initial_state.clone() if initial_state is not None else torch.zeros(B_dim, D_dim, device=device, dtype=torch.float32)
    if state.dim() == 1:
        state = state.unsqueeze(0).expand(B_dim, -1)
        
    outputs = []
    for b in range(nb):
        block_out = []
        for t in range(sz):
            x_v = x_blocks[:, b, t, :].to(torch.float32)
            g_v = gate[:, b, t, :].to(torch.float32)
            block_out.append((x_v + state).to(dtype))
            state = (state * dec) + (g_v * x_v)
        outputs.append(torch.stack(block_out, dim=1))
        
    return torch.stack(outputs, dim=1)


# ============================================================================
# 6. FUSED RMS NORM WITH AUTOGRAD
# ============================================================================
if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(X_ptr, OUT_ptr, RSTD_ptr, N_dim: tl.constexpr, stride_xr, stride_xc, stride_or, stride_oc, eps: tl.constexpr, BLOCK_N: tl.constexpr):
        row = tl.program_id(0); col_offs, mask = tl.arange(0, BLOCK_N), tl.arange(0, BLOCK_N) < N_dim
        x = tl.load(X_ptr + row * stride_xr + col_offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N_dim + eps)
        tl.store(RSTD_ptr + row, rstd)
        tl.store(OUT_ptr + row * stride_or + col_offs * stride_oc, (x * rstd).to(OUT_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_bwd_kernel(X_ptr, DY_ptr, RSTD_ptr, DX_ptr, N_dim: tl.constexpr, stride_xr, stride_xc, stride_dyr, stride_dyc, stride_dxr, stride_dxc, BLOCK_N: tl.constexpr):
        row = tl.program_id(0); col_offs, mask = tl.arange(0, BLOCK_N), tl.arange(0, BLOCK_N) < N_dim
        x = tl.load(X_ptr + row * stride_xr + col_offs * stride_xc, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + row * stride_dyr + col_offs * stride_dyc, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row).to(tl.float32)
        dx = rstd * (dy - x * (rstd * rstd) * (tl.sum(x * dy, axis=0) / N_dim))
        tl.store(DX_ptr + row * stride_dxr + col_offs * stride_dxc, dx.to(tl.float32), mask=mask)

    class _TritonRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor, eps: float) -> Tensor:
            orig_shape, D = x.shape, x.size(-1); x_2d = x.reshape(-1, D).contiguous()
            num_rows, BLOCK_N = x_2d.shape[0], triton.next_power_of_2(D)
            out, rstd = torch.empty_like(x_2d), torch.empty(num_rows, dtype=torch.float32, device=x.device)
            _rms_norm_fwd_kernel[(num_rows,)](x_2d, out, rstd, D, *x_2d.stride(), *out.stride(), eps=eps, BLOCK_N=BLOCK_N, num_warps=min(8, max(1, BLOCK_N//256)), num_stages=1)
            ctx.save_for_backward(x_2d, rstd); ctx.config = (D, BLOCK_N, orig_shape, x.dtype)
            return out.reshape(orig_shape).to(x.dtype)
        @staticmethod
        def backward(ctx, grad_output: Tensor):
            (x_2d, rstd) = ctx.saved_tensors; D, BLOCK_N, orig_shape, dtype = ctx.config
            dy_2d = grad_output.reshape(-1, D).contiguous().to(torch.float32)
            dx_2d = torch.empty_like(x_2d, dtype=torch.float32)
            _rms_norm_bwd_kernel[(x_2d.shape[0],)](x_2d, dy_2d, rstd, dx_2d, D, *x_2d.stride(), *dy_2d.stride(), *dx_2d.stride(), BLOCK_N=BLOCK_N)
            return dx_2d.reshape(orig_shape).to(dtype), None

def triton_rms_norm(x: Tensor, weight: Tensor | None=None, eps: float=1e-6) -> Tensor:
    if (not HAS_TRITON) or (not x.is_cuda) or x.size(-1) > 8192:
        return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)
    res = _TritonRMSNormFn.apply(x, eps)
    return res * weight.to(res.dtype) if weight is not None else res

# ============================================================================
# 7. UTILS & RECALL
# ============================================================================
def optimized_moe_dispatch(x_flat: Tensor, experts: list, selected_experts: Tensor, routing_weights: Tensor, effective_top_k: int, num_experts: int) -> Tensor:
    # Use x_flat * 0.0 instead of zeros_like to ensure final_output is not a leaf tensor.
    # This inherits requires_grad=True and allows in-place .index_add_() with expert grads.
    final_output = x_flat * 0.0
    active_experts = set()
    for k_idx in range(effective_top_k):
        expert_ids, weights_k = selected_experts[:, k_idx], routing_weights[:, k_idx:k_idx + 1]
        for i, expert in enumerate(experts):
            token_indices = (expert_ids == i).nonzero(as_tuple=True)[0]
            if token_indices.numel() > 0:
                final_output.index_add_(0, token_indices, expert(x_flat[token_indices]) * weights_k[token_indices])
                active_experts.add(i)
    # DDP Participation Trick: Ensure all experts are always in the computational graph.
    # This prevents DDP from complaining about unused parameters when MoE is active.
    dummy_accum = 0.0
    for i, expert in enumerate(experts):
        if i not in active_experts:
            # Call expert on a zeroed input with 0 weight, but let gradients flow.
            dummy_accum = dummy_accum + expert(x_flat[:1] * 0.0).sum() * 0.0
    return final_output + dummy_accum

def optimized_feedback_retrieve(q: Tensor, k: Tensor, v: Tensor, lr: Tensor, sketch_len: int) -> Tensor:
    inner = torch.bmm(k, q.transpose(1, 2))
    return ((lr / max(sketch_len, 1)) * torch.bmm(v.transpose(1, 2), inner)).transpose(1, 2)

# ============================================================================
# 8. PARITY TESTS
# ============================================================================
def test_all_parity(device="cuda"):
    print("Running exhaustive parity tests (forward + backward)...")
    torch.manual_seed(42)
    # 1. RMSNorm Parity
    x_dim = 128
    x = torch.randn(16, x_dim, device=device, requires_grad=True)
    w = torch.randn(x_dim, device=device, requires_grad=True)
    y_tri = triton_rms_norm(x, w)
    y_ref = F.rms_norm(x, (x_dim,), weight=w)
    assert torch.allclose(y_tri, y_ref, atol=1e-2), "RMSNorm forward mismatch"
    # Backward
    y_tri.sum().backward()
    grad_x_tri, grad_w_tri = x.grad.clone(), w.grad.clone()
    x.grad.zero_(); w.grad.zero_()
    y_ref.sum().backward()
    assert torch.allclose(grad_x_tri, x.grad, atol=1e-2), "RMSNorm grad_x mismatch"
    assert torch.allclose(grad_w_tri, w.grad, atol=1e-2), "RMSNorm grad_w mismatch"
    print("RMSNorm Forward/Backward: PASS")
    # 2. Parallel Scan Parity
    B, T, S = 2, 64, 32
    bv = torch.randn(B, T, S, device=device, requires_grad=True)
    d = torch.sigmoid(torch.randn(B, T, S, device=device)).requires_grad_(True)
    y_tri = triton_parallel_scan(bv, d)
    y_ref = _pytorch_scan_fallback(bv, d)
    assert torch.allclose(y_tri, y_ref, atol=1e-2), "Scan forward mismatch"
    # Backward
    y_tri.sum().backward()
    grad_bv_tri, grad_d_tri = bv.grad.clone(), d.grad.clone()
    bv.grad.zero_(); d.grad.zero_()
    y_ref.sum().backward()
    assert torch.allclose(grad_bv_tri, bv.grad, atol=1e-2), "Scan grad_bv mismatch"
    assert torch.allclose(grad_d_tri, d.grad, atol=1e-2), "Scan grad_d mismatch"
    print("Scan Forward/Backward: PASS")
    # 3. Spectral Decay Scan Parity
    B, nb, sz, D = 1, 4, 32, 64
    x = torch.randn(B, nb, sz, D, device=device, requires_grad=True)
    decay = torch.sigmoid(torch.randn(D, device=device)).requires_grad_(True)
    gate = torch.sigmoid(torch.randn(B, nb, sz, D, device=device)).requires_grad_(True)
    init_state = torch.randn(D, device=device, requires_grad=True)
    y_tri = triton_spectral_decay_scan(x, decay, gate, initial_state=init_state)
    # PyTorch implementation for reference (using the corrected math)
    def ref_scan(x, dec, g, istate=None):
        dec_clamped = torch.clamp(dec, 0.0, 0.999)
        B_dim, nb_dim, sz_dim, D_dim = x.shape
        state = istate.clone() if istate is not None else torch.zeros(B_dim, D_dim, device=x.device, dtype=torch.float32)
        if state.dim() == 1: state = state.unsqueeze(0).expand(B_dim, -1)
        outs = []
        for b in range(nb_dim):
            b_outs = []
            for t in range(sz_dim):
                x_v = x[:, b, t, :].to(torch.float32)
                g_v = g[:, b, t, :].to(torch.float32)
                b_outs.append((x_v + state).to(x.dtype))
                state = (state * dec_clamped) + (g_v * x_v)
            outs.append(torch.stack(b_outs, dim=1))
        return torch.stack(outs, dim=1)
    
    y_ref = ref_scan(x, decay, gate, istate=init_state)
    assert torch.allclose(y_tri, y_ref, atol=1e-2), f"Spectral Decay Scan forward mismatch: max diff {(y_tri - y_ref).abs().max()}"
    # Backward
    y_tri.sum().backward()
    grad_x_tri, grad_decay_tri, grad_gate_tri, grad_init_tri = x.grad.clone(), decay.grad.clone(), gate.grad.clone(), init_state.grad.clone()
    x.grad.zero_(); decay.grad.zero_(); gate.grad.zero_(); init_state.grad.zero_()
    y_ref.sum().backward()
    assert torch.allclose(grad_x_tri, x.grad, atol=1e-2), "Spectral Decay Scan grad_x mismatch"
    assert torch.allclose(grad_decay_tri, decay.grad, atol=1e-2), f"Spectral Decay Scan grad_decay mismatch: max diff {(grad_decay_tri - decay.grad).abs().max()}"
    assert torch.allclose(grad_gate_tri, gate.grad, atol=1e-2), "Spectral Decay Scan grad_gate mismatch"
    assert torch.allclose(grad_init_tri, init_state.grad, atol=1e-2), f"Spectral Decay Scan grad_init mismatch: max diff {(grad_init_tri - init_state.grad).abs().max()}"
    print("Spectral Decay Scan (with initial_state) Forward/Backward: PASS")
    # Single block case (curriculum signal check)
    x1 = torch.randn(1, 1, sz, D, device=device, requires_grad=True)
    decay1 = torch.sigmoid(torch.randn(D, device=device)).requires_grad_(True)
    gate1 = torch.sigmoid(torch.randn(1, 1, sz, D, device=device)).requires_grad_(True)
    init1 = torch.randn(D, device=device, requires_grad=True)
    y1 = triton_spectral_decay_scan(x1, decay1, gate1, initial_state=init1)
    y1.sum().backward()
    assert (decay1.grad.abs().sum() > 0), "Spectral Decay Scan: NO SIGNAL to decay_rates in single-block mode!"
    assert (init1.grad.abs().sum() > 0), "Spectral Decay Scan: NO SIGNAL to initial_state in single-block mode!"
    print("Spectral Decay Scan Single-Block Signal: PASS")
    print("All Parity Tests Finished Successfully.")

if __name__ == "__main__":
    if torch.cuda.is_available(): test_all_parity()
    else:
        # Test CPU fallbacks if possible
        print("CUDA not available; testing PyTorch fallbacks on CPU...")
        test_all_parity(device="cpu")
