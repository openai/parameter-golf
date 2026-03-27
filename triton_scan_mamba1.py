"""
Triton selective scan kernel for Mamba SSM.

The key insight: the sequential scan was slow because of PYTHON loop overhead
(1024 iterations × CUDA kernel launch latency). Moving the loop into a
JIT-compiled Triton kernel eliminates this entirely.

Each Triton program handles one (batch, d_inner) lane. It loops over L=1024
timesteps sequentially, maintaining d_state=8 state values in registers.
With batch*d_inner=32768 programs running in parallel, the GPU is fully utilized.

Forward:  h[t] = gate[t]*h[t-1] + dt[t]*B[t]*x[t],  y[t] = C[t]·h[t] + D*x[t]
Backward: dh[t] = C[t]*dy[t] + gate[t]*dh[t+1]  (reverse scan)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _selective_scan_fwd_kernel(
    # Pointers
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
    y_ptr,       # output
    h_all_ptr,   # save all hidden states for backward: (batch*d_inner, L, d_state)
    # Strides for (batch, L, d_inner) tensors
    stride_xb, stride_xl, stride_xd,
    # Strides for (batch, L, d_state) tensors
    stride_Bb, stride_Bl, stride_Bn,
    # Strides for h_all: (batch*d_inner, L, d_state)
    stride_hrow, stride_hl, stride_hn,
    # Dimensions
    d_inner,
    L: tl.constexpr,
    D_STATE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // d_inner
    d_idx = pid % d_inner

    # Load A for this lane: (D_STATE,)
    n_range = tl.arange(0, D_STATE)
    a = tl.load(A_ptr + d_idx * D_STATE + n_range)  # negative values
    d_val = tl.load(D_ptr + d_idx)

    # Initialize hidden state
    h = tl.zeros([D_STATE], dtype=tl.float32)

    for t in range(L):
        # Load scalar x[t] and dt[t] for this lane
        x_off = batch_idx * stride_xb + t * stride_xl + d_idx * stride_xd
        x_t = tl.load(x_ptr + x_off).to(tl.float32)
        dt_t = tl.load(dt_ptr + x_off).to(tl.float32)

        # Load B[t, :] and C[t, :] for this batch: (D_STATE,)
        B_off = batch_idx * stride_Bb + t * stride_Bl + n_range * stride_Bn
        C_off = batch_idx * stride_Bb + t * stride_Bl + n_range * stride_Bn
        b_t = tl.load(B_ptr + B_off).to(tl.float32)
        c_t = tl.load(C_ptr + C_off).to(tl.float32)

        # SSM step
        gate = tl.exp(dt_t * a)        # (D_STATE,)
        h = gate * h + dt_t * b_t * x_t  # (D_STATE,)
        y_t = tl.sum(c_t * h) + d_val * x_t  # scalar

        # Store output
        tl.store(y_ptr + x_off, y_t)

        # Save h for backward
        h_off = pid * stride_hrow + t * stride_hl + n_range * stride_hn
        tl.store(h_all_ptr + h_off, h)


# ---------------------------------------------------------------------------
# Backward kernel
#
# Strategy: two passes within one kernel launch.
#   Pass 1 (forward): recompute h[t] and save h[t-1] into a local variable
#           Also precompute gate[t] and store per-timestep values we need.
#   Pass 2 (backward): reverse scan for dh, compute all gradients.
#
# Since we can't store L*D_STATE in registers, we use the h_all buffer
# that was saved during the forward pass.
# ---------------------------------------------------------------------------
@triton.jit
def _selective_scan_bwd_kernel(
    # Forward inputs
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
    # Saved hidden states from forward: (batch*d_inner, L, d_state)
    h_all_ptr,
    # Upstream gradient
    dy_ptr,
    # Gradient outputs
    dx_ptr, ddt_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr,
    # Strides for (batch, L, d_inner) tensors
    stride_xb, stride_xl, stride_xd,
    # Strides for (batch, L, d_state) tensors
    stride_Bb, stride_Bl, stride_Bn,
    # Strides for h_all: (batch*d_inner, L, d_state)
    stride_hrow, stride_hl, stride_hn,
    # Dimensions
    d_inner,
    L: tl.constexpr,
    D_STATE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // d_inner
    d_idx = pid % d_inner

    n_range = tl.arange(0, D_STATE)
    a = tl.load(A_ptr + d_idx * D_STATE + n_range)
    d_val = tl.load(D_ptr + d_idx)

    # Accumulators for dD and dA (summed over L for this (batch, d_inner) lane)
    dD_acc = 0.0
    dA_acc = tl.zeros([D_STATE], dtype=tl.float32)

    # Backward scan: process timesteps in reverse
    # dh accumulates the gradient flowing backward through the recurrence:
    #   dh_full[t] = C[t]*dy[t] + gate[t]*dh_full[t+1]
    #
    # At each step, we propagate: dh <- gate[t] * dh_full[t]
    # so that at the next iteration (t-1), dh = gate[t]*dh_full[t].
    dh = tl.zeros([D_STATE], dtype=tl.float32)

    # Track h_prev as we go backward.
    # We load h[t] at each step and keep it for the NEXT backward step
    # (which processes t-1 and needs h[(t-1)-1] = h[t-2]).
    # Actually that's not right — we need h[t-1] at step t.
    #
    # Solution: load h[t] from h_all for each t, and for h[t-1] also
    # load from h_all. To handle t=0, we load h[0] (safe index) and
    # multiply by 0 using the h_prev tracker initialized to zeros.
    #
    # Cleaner: run a forward pass to recompute h, storing h_prev at each step.
    # But that requires O(L) storage. Instead, just do two loads from h_all.
    #
    # Simplest safe approach: track h_prev from the forward direction.
    # We run a FORWARD loop first just to cache h[t-1] values? No, too complex.
    #
    # Just load h[t] and h[t-1] from global memory. For t=0, h[-1]=0.
    # Use t_prev = max(t-1, 0) and a mask.

    for t_rev in range(L):
        t = L - 1 - t_rev

        # Load forward values for this timestep
        x_off = batch_idx * stride_xb + t * stride_xl + d_idx * stride_xd
        x_t = tl.load(x_ptr + x_off).to(tl.float32)
        dt_t = tl.load(dt_ptr + x_off).to(tl.float32)
        dy_t = tl.load(dy_ptr + x_off).to(tl.float32)

        B_off = batch_idx * stride_Bb + t * stride_Bl + n_range * stride_Bn
        C_off = batch_idx * stride_Bb + t * stride_Bl + n_range * stride_Bn
        b_t = tl.load(B_ptr + B_off).to(tl.float32)
        c_t = tl.load(C_ptr + C_off).to(tl.float32)

        # Load saved h[t]
        h_off_t = pid * stride_hrow + t * stride_hl + n_range * stride_hn
        h_t = tl.load(h_all_ptr + h_off_t)

        # Load h[t-1]: for t=0, load h[0] (address-safe) then zero it out.
        # For t>0, load h[t-1] normally.
        # We use max(t-1, 0) for the address, which equals 0 when t=0 and t-1 otherwise.
        t_prev = tl.maximum(t - 1, 0)
        h_off_prev = pid * stride_hrow + t_prev * stride_hl + n_range * stride_hn
        h_prev = tl.load(h_all_ptr + h_off_prev)
        # When t=0: t_prev=0 means we loaded h[0], but h[-1] should be 0.
        # Mask it out. t_rev == L-1 when t == 0.
        h_prev = tl.where(t > 0, h_prev, 0.0)

        # Recompute gate
        gate_t = tl.exp(dt_t * a)  # (D_STATE,)

        # dh_full[t] = C[t]*dy[t] + dh  (where dh = gate[t+1]*dh_full[t+1])
        dh_full = c_t * dy_t + dh  # (D_STATE,)

        # ---- Local gradients ----
        # From y[t] = C[t]·h[t] + D*x[t]:
        # dC[batch, t, n] = sum_d(dy[batch, t, d] * h[batch, t, d, n])
        # Each program contributes one d_idx, so atomic_add across d_inner lanes
        dC_t = dy_t * h_t  # (D_STATE,)
        tl.atomic_add(dC_ptr + C_off, dC_t)

        # From h[t] = gate[t]*h[t-1] + dt[t]*B[t]*x[t]:
        d_input = dh_full              # gradient w.r.t. the additive input term
        d_gate = dh_full * h_prev      # gradient w.r.t. the multiplicative gate

        # dx[t] = D*dy[t] + sum_n(d_input[n] * dt[t] * B[t,n])
        dx_t = d_val * dy_t + tl.sum(d_input * dt_t * b_t)
        tl.store(dx_ptr + x_off, dx_t)

        # ddt[t]: from input term (dt*B*x) and gate term (exp(dt*A))
        # gate = exp(dt*A) => d_gate/d_dt = A * gate
        ddt_t = tl.sum(d_input * b_t * x_t) + tl.sum(d_gate * a * gate_t)
        tl.store(ddt_ptr + x_off, ddt_t)

        # dB[batch, t, n] = sum_d(d_input[d, n] * dt[t, d] * x[t, d])
        # Each program contributes one d_idx, so atomic_add across d_inner lanes
        dB_t = d_input * dt_t * x_t  # (D_STATE,)
        tl.atomic_add(dB_ptr + B_off, dB_t)

        # dA[d,n] += d_gate[n] * dt[t] * gate[t,n]
        # gate = exp(dt*A) => d_gate/d_A = dt * gate
        dA_acc += d_gate * dt_t * gate_t

        # dD += dy[t] * x[t]
        dD_acc += dy_t * x_t

        # Propagate dh backward for the next iteration (t-1)
        dh = gate_t * dh_full

    # Write accumulated gradients (atomic add since multiple batch elements contribute)
    tl.atomic_add(dD_ptr + d_idx, dD_acc)
    tl.atomic_add(dA_ptr + d_idx * D_STATE + n_range, dA_acc)


# ---------------------------------------------------------------------------
# Autograd wrapper
# ---------------------------------------------------------------------------
class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D):
        batch, L, d_inner = x.shape
        d_state = A.shape[1]

        # Ensure contiguous
        x = x.contiguous()
        dt = dt.contiguous()
        A = A.contiguous().float()
        B = B.contiguous()
        C = C.contiguous()

        y = torch.empty_like(x)
        # Buffer to save hidden states for backward
        h_all = torch.empty(batch * d_inner, L, d_state, device=x.device, dtype=torch.float32)

        grid = (batch * d_inner,)
        _selective_scan_fwd_kernel[grid](
            x, dt, A, B, C, D,
            y, h_all,
            x.stride(0), x.stride(1), x.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            h_all.stride(0), h_all.stride(1), h_all.stride(2),
            d_inner, L, d_state,
        )

        ctx.save_for_backward(x, dt, A, B, C, D, h_all)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, dt, A, B, C, D, h_all = ctx.saved_tensors
        batch, L, d_inner = x.shape
        d_state = A.shape[1]

        dy = dy.contiguous()
        dx = torch.empty_like(x)
        ddt = torch.empty_like(dt)
        dA = torch.zeros_like(A)  # (d_inner, d_state), accumulated via atomic_add
        dB = torch.zeros_like(B)  # accumulated via atomic_add across d_inner
        dC = torch.zeros_like(C)  # accumulated via atomic_add across d_inner
        dD = torch.zeros(d_inner, device=x.device, dtype=torch.float32)

        grid = (batch * d_inner,)
        _selective_scan_bwd_kernel[grid](
            x, dt, A, B, C, D,
            h_all,
            dy,
            dx, ddt, dA, dB, dC, dD,
            x.stride(0), x.stride(1), x.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            h_all.stride(0), h_all.stride(1), h_all.stride(2),
            d_inner, L, d_state,
        )

        return dx, ddt, dA, dB, dC, dD


def selective_scan(x, dt, A, B, C, D):
    """Drop-in replacement for the sequential Python selective_scan.

    Args:
        x:  (batch, L, d_inner)  - input after conv1d + silu
        dt: (batch, L, d_inner)  - discretization timestep (after softplus)
        A:  (d_inner, d_state)   - state transition (negative values)
        B:  (batch, L, d_state)  - input projection
        C:  (batch, L, d_state)  - output projection
        D:  (d_inner,)           - skip connection

    Returns:
        y:  (batch, L, d_inner)
    """
    return SelectiveScanFn.apply(x, dt, A, B, C, D)
