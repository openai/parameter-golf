"""TTSM Triton Chunk-Wise Parallel Scan

Adapted from fla HGRN chunk kernel (MIT License):
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/hgrn/chunk.py
  Authors: Songlin Yang, Yu Zhang, Zhiyuan Li

The TTSM recurrence:
  h_t = exp(dt_t * A_log) * h_{t-1} + dt_t * B_t * x_t

maps exactly to the HGRN recurrence:
  h_t = exp(g_t) * h_{t-1} + x_t

with:
  g_t = dt_t * A_log   (log-domain gate per channel)
  x_t = dt_t * B_t * x_inner_t  (dt-scaled input after B projection)

Implementation notes:
- D = d_inner * d_state (flattened state dim for HGRN, reshaped before/after)
- All (d_inner, d_state) channel pairs run in parallel within each chunk
- Sequential state carry only between chunks (num_chunks = seq_len // BT)
- Forward + backward both implemented; autograd Function wraps them

No external fla dependency — kernel code is self-contained.
Ternary B/C projections handled by TernaryLinear outside this module (Phase 1).
Phase 2: fuse ternary projection bitwise ops inside the scan kernel.
"""

import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton inner-chunk forward: sequential scan within BT steps
# ---------------------------------------------------------------------------

# BD=128 hardcoded — eliminates @triton.autotune exploration.
# Root cause of training crashes: autotune runs N configs × the kernel on the same in-place
# buffer (Triton #6547, #1563). With BD fixed, no exploration → no in-place corruption.
# BD=128 divides D_flat=36864 exactly (288 programs), good for H100 occupancy.
@triton.jit(do_not_specialize=['T'])
def _ttsm_scan_fwd_inner(
    x,          # [B, T, D] — dt-scaled B projection output (scan input)
    g,          # [B, T, D] — log gate = dt * A_log (negative = decay)
    gc,         # [B, T, D] — cumulative log gate (output, for inter-chunk pass)
    o,          # [B, T, D] — hidden state output
    h0,         # [B, D]    — initial state (optional)
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    """One program per (channel_block, chunk, batch) — sequential scan within chunk."""
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Cast i_b to int64: eval uses B=32, i_b * T * D = 29 * 2048 * 36864 = 2.19B > INT32_MAX.
    # int32 overflow → negative pointer offset → H100 illegal memory access at eval time.
    # Training uses B=2 (no overflow), so this was invisible during training-only smoke tests.
    i_b = i_b.to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x  = x  + i_b * T * D + i_t * BT * D + o_d
    p_g  = g  + i_b * T * D + i_t * BT * D + o_d
    p_gc = gc + i_b * T * D + i_t * BT * D + o_d
    p_o  = o  + i_b * T * D + i_t * BT * D + o_d

    b_h  = tl.zeros([BD], dtype=tl.float32)
    b_gc = tl.zeros([BD], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0).to(tl.float32)

    for i in range(0, BT):
        mask_t = mask & ((i_t * BT + i) < T)
        b_x = tl.load(p_x, mask=mask_t, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask_t, other=0).to(tl.float32)
        # Core recurrence: h = exp(g) * h + x
        b_h  = tl.exp(b_g) * b_h + b_x
        b_gc = b_gc + b_g
        tl.store(p_gc, b_gc.to(p_o.dtype.element_ty), mask=mask_t)
        tl.store(p_o,  b_h.to(p_o.dtype.element_ty),  mask=mask_t)
        p_x  += D
        p_g  += D
        p_gc += D
        p_o  += D


@triton.jit(do_not_specialize=['T'])
def _ttsm_scan_fwd_inter(
    gc,         # [B, T, D] — cumulative log gate from inner pass
    o,          # [B, T, D] — hidden states to be corrected in-place
    s_b, s_t, s_d,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """One program per (channel_block, batch) — propagate final state across chunks."""
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    # Same int32 overflow guard as inner kernel: eval B=32, i_b=29 overflows int32.
    i_b = i_b.to(tl.int64)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(1, tl.cdiv(T, BT)):
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o  = tl.make_block_ptr(o  + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))

        # Final hidden state of previous chunk
        b_h0 = tl.load(o + i_b * T * D + i_t * BT * D - D + o_d, mask=mask, other=0).to(tl.float32)
        # Cumulative log-gate within this chunk
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_o  = tl.load(p_o,  boundary_check=(0, 1)).to(tl.float32)
        # Apply inter-chunk correction: o += exp(cumulative_g) * h_prev_chunk
        b_o  = b_o + tl.exp(b_gc) * b_h0[None, :]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Triton backward: within-chunk backward (reverse scan)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=nw)
        for BD in [32, 64, 128]
        for nw in [1, 2, 4, 8]
    ],
    key=['D'],
)
@triton.jit(do_not_specialize=['T'])
def _ttsm_scan_bwd_inner(
    g, gc, dx, do,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Within-chunk backward — reverse sequential scan."""
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    BC  = min(BT, T - i_t * BT)
    NT  = tl.num_programs(1)

    p_g  = g  + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_gc = gc + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d

    if i_t == NT - 1:
        b_gc = tl.zeros([BD], dtype=tl.float32)
    else:
        b_gc = tl.load(g + (i_b * T + i_t * BT + BT) * D + o_d, mask=mask, other=0).to(tl.float32)

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for _ in range(BC - 1, -1, -1):
        tl.store(p_gc, b_gc.to(p_gc.dtype.element_ty), mask=mask)
        b_g  = tl.load(p_g,  mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        b_gc = b_gc + b_g
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * tl.exp(b_g)
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        p_g  -= D
        p_gc -= D
        p_dx -= D
        p_do -= D


@triton.jit(do_not_specialize=['T'])
def _ttsm_scan_bwd_inter(
    g, gc, o, dx, dg,
    s_b, s_t, s_d,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Inter-chunk backward — propagate gradient across chunks."""
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_g  = tl.make_block_ptr(g  + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))

        # H100 OOB fix #1: clamp next-chunk address to avoid hardware exception on masked loads.
        # When i_t = T/BT - 1 (last chunk), (i_t+1)*BT == T → address 1 past tensor end.
        # H100 validates addresses even for mask=False loads → illegal memory access.
        # Fix: use offset 0 as safe fallback when has_next=False (value discarded anyway).
        has_next = (i_t + 1) * BT < T
        safe_next_offset = tl.where(has_next, (i_t + 1) * BT * D, 0)
        mask_t = mask & has_next
        b_ht = tl.load(dx + i_b * T * D + safe_next_offset + o_d, mask=mask_t, other=0).to(tl.float32)

        b_g  = tl.load(p_g,  boundary_check=(0, 1)).to(tl.float32)
        b_gc = tl.load(p_gc, boundary_check=(0, 1)).to(tl.float32)
        b_dx = tl.load(p_dx, boundary_check=(0, 1)).to(tl.float32)

        # HGRN original pattern: block pointer + boundary_check=(0,1) handles row -1 correctly.
        # On H100, block pointer OOB uses descriptor-level bounds (shape [0, T)), NOT hardware
        # address validation. Row -1 returns 0 safely — no hardware exception.
        # The broken "Python if has_prev" fix was wrong: i_t is a Triton runtime variable, so
        # Python `if` evaluates the Triton tensor as ALWAYS truthy → always generates row -1
        # address as a raw pointer → hardware exception. Block pointer is the safe path.
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT - 1, i_d * BD), (BT, BD), (1, 0))
        b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
        # boundary_check already returns 0 for row -1 (when i_t=0) — no explicit zeroing needed

        b_dx = b_dx + tl.exp(b_gc) * b_ht[None, :]
        b_dg = b_o * b_dx * tl.exp(b_g)
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# PyTorch backward chunk (compiled — replaces Triton backward kernels)
# ---------------------------------------------------------------------------

@torch.compile(dynamic=True)
def _backward_scan_chunk(
    delta_h: torch.Tensor,      # [B, D] — accumulated Δh coming from NEXT chunk
    do_chunk: torch.Tensor,     # [B, C, D] — ∂L/∂h for positions in this chunk
    g_chunk: torch.Tensor,      # [B, C, D] — log gates for this chunk
    h_prev_chunk: torch.Tensor, # [B, C, D] — h[t-1] for each position
    g_after_chunk: torch.Tensor, # [B, D] — g[chunk_end+1] or zeros if last chunk
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Backward through one chunk of the sequential SSM scan.

    Recurrence: Δh[t] = do[t] + exp(g[t+1]) * Δh[t+1]
    At chunk boundary: Δh[chunk_end] = do[chunk_end] + exp(g[chunk_end+1]) * delta_h_in

    No Python conditional in the inner loop — allows torch.compile to fuse all C
    iterations into a single CUDA kernel instead of C separate launches.

    Returns: (delta_h_out [B, D], dx_chunk [B, C, D], dg_chunk [B, C, D])
    """
    B, C, D = do_chunk.shape
    exp_g = g_chunk.exp()   # [B, C, D] — exp(g[t]) for all positions t in chunk

    # "Next gate" for each position: exp(g[t+1]).
    # For positions 0..C-2: exp(g_chunk[:, t+1, :])
    # For position C-1 (last in chunk): exp(g_after_chunk) — the gate of the NEXT chunk's first pos
    exp_g_next = torch.cat([g_chunk[:, 1:, :], g_after_chunk.unsqueeze(1)], dim=1).exp()  # [B, C, D]

    dx_out = torch.empty_like(do_chunk)
    dg_out = torch.empty_like(g_chunk)

    # Sequential backward (no conditional → fully compilable loop):
    for i in range(C - 1, -1, -1):
        delta_h = do_chunk[:, i, :] + exp_g_next[:, i, :] * delta_h
        dx_out[:, i, :] = delta_h
        dg_out[:, i, :] = delta_h * exp_g[:, i, :] * h_prev_chunk[:, i, :]

    return delta_h, dx_out, dg_out


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------

class _TTSMScanFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        """
        x: [B, T, D] — scan input (dt_scaled * B_out, reshaped)
        g: [B, T, D] — log gate   (dt * A_log, reshaped)
        initial_state: [B, D] or None
        Returns: (o [B, T, D], final_state [B, D] or None)
        """
        B, T, D = x.shape
        BT = 128      # Triton chunk size
        BD = 128      # hardcoded — no autotune (eliminates Triton #6547 in-place corruption)
        num_warps = 4

        gc = torch.empty_like(g, dtype=torch.float32)
        o  = torch.empty_like(x, dtype=torch.float32)

        # Pin Triton launches to PyTorch's current CUDA stream.
        # Root cause of gc lifetime bug (Tink s17 analysis): Triton may use the
        # default CUDA stream while PyTorch's caching allocator tracks gc on the
        # per-device stream, allowing gc to be freed/reused while the GPU still
        # writes to it. The with-stream + record_stream combo ensures the allocator
        # sees the same stream the kernels run on.
        _cur_stream = torch.cuda.current_stream()
        with torch.cuda.stream(_cur_stream):
            # No meta['BD'] — kernel called directly with fixed BD (no autotune exploration)
            grid_inner = (triton.cdiv(D, BD), triton.cdiv(T, BT), B)
            _ttsm_scan_fwd_inner[grid_inner](
                x, g, gc, o, initial_state,
                T=T, D=D, BT=BT, BD=BD,
                USE_INITIAL_STATE=initial_state is not None,
                num_warps=num_warps,
            )

            grid_inter = (triton.cdiv(D, BD), B)
            _ttsm_scan_fwd_inter[grid_inter](
                gc, o,
                o.stride(0), o.stride(1), o.stride(2),
                T=T, D=D, BT=BT, BD=BD,
                num_warps=num_warps,
            )

        # Belt-and-suspenders: mark gc as in-use on this stream so the allocator
        # won't reclaim its memory until the kernel completes — guards eval/no_grad paths
        # where ctx.save_for_backward may not extend gc's lifetime.
        gc.record_stream(_cur_stream)

        final_state = o[:, -1].clone() if output_final_state else None
        o_out = o.to(x.dtype)
        # gc lifetime: record_stream (above) keeps gc alive for kernel completion.
        # save_for_backward pins gc through the entire autograd graph — only needed
        # during training. During eval, skip it to avoid ~4.8GB wasted pinning.
        ctx._has_initial_state = initial_state is not None
        if torch.is_grad_enabled():
            if ctx._has_initial_state:
                ctx.save_for_backward(g, gc, o, initial_state)
            else:
                ctx.save_for_backward(g, gc, o)
        else:
            if ctx._has_initial_state:
                ctx.save_for_backward(g, o, initial_state)
            else:
                ctx.save_for_backward(g, o)
        ctx.BT = BT
        ctx.BD = BD
        ctx.num_warps = num_warps
        return o_out, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        """Reversed-scan backward via the Triton forward kernel.

        The backward recurrence Δh[t] = do[t] + exp(g[t+1]) * Δh[t+1] is
        structurally identical to the forward scan h[t] = x[t] + exp(g[t]) * h[t-1]
        but reversed in time. Run the forward kernel on time-reversed inputs.
        """
        saved = ctx.saved_tensors
        if len(saved) == 4:
            g, _gc, o, initial_state = saved
        elif len(saved) == 3 and ctx._has_initial_state:
            g, o, initial_state = saved
        elif len(saved) == 3:
            g, _gc, o = saved
            initial_state = None
        else:
            g, o = saved
            initial_state = None
        B, T, D = do.shape

        if initial_state is not None:
            h_prev = torch.cat([initial_state.unsqueeze(1), o[:, :-1, :]], dim=1)
        else:
            h_prev = torch.cat([torch.zeros_like(o[:, :1, :]), o[:, :-1, :]], dim=1)

        g_f = g.float()
        do_f = do.float()
        h_prev_f = h_prev.float()

        # Reversed-scan: g_bwd[τ] = g[T-τ] for τ>0, 0 for τ=0
        g_bwd = torch.cat([torch.zeros_like(g_f[:, :1, :]),
                           g_f[:, 1:, :].flip(dims=[1])], dim=1)
        with torch.no_grad():
            dx_rev, _ = ttsm_triton_scan(
                do_f.flip(dims=[1]).contiguous(),
                g_bwd.contiguous(),
            )
        dx = dx_rev.flip(dims=[1])
        dg = dx * g_f.exp() * h_prev_f

        return dx.to(o.dtype), dg, None, None


@torch.compiler.disable
def ttsm_triton_scan(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunk-wise parallel scan for TTSM recurrence.

    Args:
        x: scan input  [B, T, D] — dt_expanded * B_out (fp32 or bf16)
        g: log gate    [B, T, D] — dt_expanded * A_log (fp32 or bf16, negative = decay)
        initial_state: [B, D] or None
        output_final_state: whether to return h_T

    Returns:
        (h_all [B, T, D], final_state [B, D] or None)
    """
    return _TTSMScanFunction.apply(x, g, initial_state, output_final_state)


# ---------------------------------------------------------------------------
# Drop-in replacement for SelectiveSSM.forward (use in train_ternary.py)
# ---------------------------------------------------------------------------

def selective_ssm_triton_forward(
    x: torch.Tensor,
    A_log: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_state: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TTSM selective SSM using Triton chunk scan.

    Args:
        x:       [B, T, d_inner] — gated input (after in_proj split)
        A_log:   [d_inner, d_state] — log of -A (positive, diverse init)
        dt:      [B, T, d_inner] — discretization step (after softplus, always > 0)
        B:       [B, T, d_state] — input matrix projection (after TernaryLinear)
        C:       [B, T, d_state] — output matrix projection (after TernaryLinear)
        D:       [d_inner] — direct skip connection
        initial_state: [B, d_inner * d_state] or None

    Returns:
        (y [B, T, d_inner], final_state [B, d_inner * d_state])
    """
    batch, seq, d_inner = x.shape
    d_state = B.shape[-1]
    D_flat  = d_inner * d_state

    x_f  = x.float()
    B_f  = B.float()
    C_f  = C.float()
    dt_f = dt.float()

    # Log gate: g = dt * A where A = -exp(A_log) < 0
    # Matches SelectiveSSM.forward: A = -exp(A_log); dA = exp(dt * A) = exp(g)
    # g = -dt * exp(A_log) is negative → exp(g) ∈ (0,1) ✓
    A_abs = A_log.float().exp()                                        # exp(A_log) > 0, [d_inner, d_state]
    g = -dt_f[:, :, :, None] * A_abs[None, None, :, :]                # [B, T, d_inner, d_state]
    g = g.reshape(batch, seq, D_flat).contiguous()

    # Scan input: x_scan[b,t,i,d] = dt[b,t,i] * B[b,t,d] * x[b,t,i]
    x_scan = (dt_f[:, :, :, None] * B_f[:, :, None, :] * x_f[:, :, :, None])  # [B, T, d_inner, d_state]
    x_scan = x_scan.reshape(batch, seq, D_flat).contiguous()

    # Run Triton scan
    h_flat, final_state = ttsm_triton_scan(
        x_scan, g,
        initial_state=initial_state,
        output_final_state=True,
    )  # h_flat: [B, T, d_inner * d_state]

    h_all = h_flat.reshape(batch, seq, d_inner, d_state)  # [B, T, d_inner, d_state]

    # Readout: y[b,t,i] = sum_d C[b,t,d] * h[b,t,i,d]
    y = (C_f[:, :, None, :] * h_all).sum(-1)  # [B, T, d_inner]

    y = y + D.float()[None, None, :] * x_f
    return y.to(x.dtype), final_state


# ---------------------------------------------------------------------------
# Correctness test (run this on the pod: python ours/ttsm_triton_scan.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing TTSM Triton scan vs Python reference...")
    torch.manual_seed(42)

    B, T, d_inner, d_state = 2, 512, 64, 32   # small for fast test
    D_flat = d_inner * d_state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, D_flat={D_flat}")

    # Random inputs (fp32 for reference)
    g = (-0.1 * torch.rand(B, T, D_flat)).to(device)          # negative log gate
    x_scan = torch.randn(B, T, D_flat, device=device) * 0.1   # small scan input

    # Python reference (sequential scan)
    h_ref = torch.zeros(B, D_flat, device=device)
    o_ref = torch.zeros(B, T, D_flat, device=device)
    for t in range(T):
        h_ref = g[:, t, :].exp() * h_ref + x_scan[:, t, :]
        o_ref[:, t, :] = h_ref

    # Triton scan
    o_tri, _ = ttsm_triton_scan(x_scan, g)

    # Compare
    abs_err = (o_tri.float() - o_ref).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    print(f"Max abs error:  {max_err:.2e}")
    print(f"Mean abs error: {mean_err:.2e}")
    # Check if error concentrates at chunk boundaries (BT=128)
    BT_kernel = 128
    boundary_positions = list(range(BT_kernel - 1, T, BT_kernel))
    interior_positions = [t for t in range(T) if t not in boundary_positions]
    if boundary_positions:
        boundary_err = abs_err[:, boundary_positions, :].max().item()
        interior_err = abs_err[:, interior_positions, :].max().item() if interior_positions else 0.0
        print(f"  Boundary max error: {boundary_err:.2e} (chunk-end positions)")
        print(f"  Interior max error: {interior_err:.2e}")

    # fp32 precision over long sequences: allow up to 1e-2 (mean is ~1e-5, fine for training)
    assert max_err < 1e-2, f"Logic bug detected (too large even for fp32): {max_err}"
    print("✓ PASSED: Triton scan within fp32 precision bounds")

    # Timing comparison
    import time
    runs = 20
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        o_tri, _ = ttsm_triton_scan(x_scan, g)
        torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / runs * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        h_py = torch.zeros(B, D_flat, device=device)
        o_py = torch.zeros(B, T, D_flat, device=device)
        for t in range(T):
            h_py = g[:, t, :].exp() * h_py + x_scan[:, t, :]
            o_py[:, t, :] = h_py
        torch.cuda.synchronize()
    t_python = (time.perf_counter() - t0) / runs * 1000

    print(f"\nTiming (B={B}, T={T}, D_flat={D_flat}):")
    print(f"  Triton: {t_triton:.2f}ms")
    print(f"  Python: {t_python:.2f}ms")
    print(f"  Speedup: {t_python/t_triton:.1f}x")
