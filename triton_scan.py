"""
Mamba-2 SSD (State Space Duality) chunk-parallel selective scan.

Instead of scanning L=1024 steps sequentially, we split the sequence into
C = L/chunk_size chunks and:
  1. Compute intra-chunk outputs via parallel matmul (O(chunk^2) but on tensor cores)
  2. Compute chunk boundary states via parallel matmul
  3. Run a short sequential scan over C chunk states (16 steps instead of 1024)
  4. Add the cross-chunk state contribution to each chunk's output

Key architectural change from Mamba-1: A is a scalar per head (not diagonal per channel),
enabling the state-space-to-attention duality that makes this work.

Reference: Dao & Gu, "Transformers are SSMs" (Mamba-2), 2024.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def segsum(x: Tensor) -> Tensor:
    """Stable segment sum in log-space.

    For a vector x of length T, computes a lower-triangular matrix where
    entry [i,j] = sum(x[j+1:i+1]) for i >= j, and -inf otherwise.
    This represents cumulative decay factors between positions.

    Args:
        x: (..., T) log-space decay values
    Returns:
        (..., T, T) lower-triangular segment sums
    """
    T = x.size(-1)
    x = x[..., :, None].expand(*x.shape, T)  # (..., T, T) — broadcast x across last dim
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0.0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask2 = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask2, -torch.inf)
    return x_segsum


def ssd_chunk_scan(
    x: Tensor,    # (batch, L, nheads, headdim)
    A: Tensor,    # (batch, L, nheads) — log-space decay (dt * A, already negative)
    B: Tensor,    # (batch, L, nheads, d_state)
    C: Tensor,    # (batch, L, nheads, d_state)
    D: Tensor,    # (nheads,) — skip connection
    chunk_size: int = 64,
) -> Tensor:
    """Chunk-parallel SSD scan (pure PyTorch, no custom Triton kernels).

    This is the core Mamba-2 algorithm. By restricting A to scalar-per-head,
    the SSM recurrence factors into matmul-friendly operations that utilize
    tensor cores.

    Returns:
        y: (batch, L, nheads, headdim)
    """
    batch, L, nheads, headdim = x.shape
    d_state = B.shape[-1]

    # Pad sequence to multiple of chunk_size if needed
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    L_padded = x.shape[1]
    nchunks = L_padded // chunk_size

    # Reshape into chunks: (batch, nchunks, chunk_size, ...)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    A = A.reshape(batch, nchunks, chunk_size, nheads)
    B = B.reshape(batch, nchunks, chunk_size, nheads, d_state)
    C = C.reshape(batch, nchunks, chunk_size, nheads, d_state)

    # ===== Step 1: Intra-chunk output (parallel across chunks) =====
    # Compute the lower-triangular decay matrix within each chunk
    # A is (batch, nchunks, chunk_size, nheads) — already in log space
    # segsum gives (batch, nchunks, chunk_size, chunk_size, nheads) decay factors
    A_chunked = A.permute(0, 1, 3, 2)  # (batch, nchunks, nheads, chunk_size)
    L_mask = torch.exp(segsum(A_chunked))  # (batch, nchunks, nheads, chunk_size, chunk_size)

    # Intra-chunk attention-like computation:
    # Y_diag[b,c,t,h,p] = sum_s sum_n C[b,c,t,h,n] * L[b,c,h,t,s] * B[b,c,s,h,n] * X[b,c,s,h,p]
    # This is: C^T @ (L_mask * (B @ X^T))^T, but let's use einsum for clarity

    # BX = B^T X: (batch, nchunks, nheads, d_state, headdim) per (chunk_pos_s)
    # Actually we need per-position: B[s] * X[s] then weighted sum with L_mask[t,s]
    # More precisely:
    #   For each output position t in chunk:
    #     y[t] = sum_s L[t,s] * (sum_n C[t,n] * B[s,n]) * X[s]
    #   The inner sum_n is a dot product, giving a scalar attention weight per (t,s)
    #   Then weighted sum of X[s]

    # Compute "attention scores": CB[t,s] = sum_n C[t,n] * B[s,n]
    # C: (batch, nchunks, chunk_size_t, nheads, d_state)
    # B: (batch, nchunks, chunk_size_s, nheads, d_state)
    CB = torch.einsum('bcthn,bcshn->bchts', C, B)  # (batch, nchunks, nheads, chunk_t, chunk_s)

    # Apply causal decay mask
    CB_masked = CB * L_mask  # (batch, nchunks, nheads, chunk_t, chunk_s)

    # Weighted sum of X
    # x: (batch, nchunks, chunk_size, nheads, headdim)
    y_diag = torch.einsum('bchts,bcshp->bcthp', CB_masked, x)  # (batch, nchunks, chunk_t, nheads, headdim)

    # ===== Step 2: Chunk boundary states =====
    # For each chunk, compute the state at the end assuming zero initial state
    # decay_states[s] = exp(sum of A from position s+1 to end of chunk)
    # states[c] = sum_s decay_states[s] * B[s]^T * X[s]

    # Cumulative sum of A within each chunk (from start to each position)
    A_cumsum = torch.cumsum(A_chunked, dim=-1)  # (batch, nchunks, nheads, chunk_size)

    # Decay from each position to end of chunk = exp(total_A - cumsum_A[s])
    A_total = A_cumsum[..., -1:]  # (batch, nchunks, nheads, 1)
    decay_states = torch.exp(A_total - A_cumsum)  # (batch, nchunks, nheads, chunk_size)

    # states[c] = sum_s decay[s] * outer(B[s], X[s])
    # B: (batch, nchunks, chunk_size, nheads, d_state)
    # x: (batch, nchunks, chunk_size, nheads, headdim)
    # decay_states: (batch, nchunks, nheads, chunk_size)
    states = torch.einsum('bchs,bcshn,bcshp->bchpn', decay_states, B, x)
    # states: (batch, nchunks, nheads, headdim, d_state)

    # ===== Step 3: Inter-chunk state passing (short sequential scan) =====
    # Decay between consecutive chunks
    chunk_decay = torch.exp(A_total.squeeze(-1))  # (batch, nchunks, nheads)

    # Sequential scan over nchunks (only 16 steps for L=1024, chunk=64)
    global_states = torch.zeros(batch, nchunks, nheads, headdim, d_state,
                                device=x.device, dtype=x.dtype)
    running_state = torch.zeros(batch, nheads, headdim, d_state,
                                device=x.device, dtype=x.dtype)
    for c in range(nchunks):
        global_states[:, c] = running_state + states[:, c]
        if c < nchunks - 1:
            running_state = chunk_decay[:, c, :, None, None] * global_states[:, c]

    # ===== Step 4: Cross-chunk state contribution =====
    # For each position t in chunk c, add contribution from global_state[c-1]
    # decayed to position t

    # Decay from chunk start to each position within chunk
    state_decay = torch.exp(A_cumsum)  # (batch, nchunks, nheads, chunk_size)

    # Use global_states shifted by 1 (state entering each chunk)
    # For chunk 0, the entering state is 0 (no previous chunk)
    initial_states = torch.zeros_like(global_states[:, :1])
    prev_states = torch.cat([initial_states, global_states[:, :-1]], dim=1)
    # prev_states: (batch, nchunks, nheads, headdim, d_state)
    # We need to decay prev_state by chunk_decay to get the state entering the chunk
    # Actually, global_states already includes the decay. Let me reconsider.

    # global_states[c] is the accumulated state at the END of chunk c (including chunk c's contribution)
    # What enters chunk c is: global_states[c-1] decayed by chunk_decay[c-1]?
    # No — running_state before adding states[c] is what enters chunk c.
    # Let me redo this with explicit entering states.

    # Redo Step 3 more carefully
    entering_states = torch.zeros(batch, nchunks, nheads, headdim, d_state,
                                  device=x.device, dtype=x.dtype)
    running_state = torch.zeros(batch, nheads, headdim, d_state,
                                device=x.device, dtype=x.dtype)
    for c in range(nchunks):
        entering_states[:, c] = running_state
        # State at end of chunk c = entering_state decayed through chunk + chunk's own contribution
        end_state = chunk_decay[:, c, :, None, None] * running_state + states[:, c]
        running_state = end_state

    # Now entering_states[:, c] is the hidden state entering chunk c
    # For position t within chunk c, decay from chunk start to position t:
    # h_contribution[t] = exp(cumsum_A[t]) * entering_states[c]
    # y_off[t] = C[t] . h_contribution[t]

    # C: (batch, nchunks, chunk_size, nheads, d_state)
    # entering_states: (batch, nchunks, nheads, headdim, d_state)
    # state_decay: (batch, nchunks, nheads, chunk_size)

    # h_at_t = state_decay[t] * entering_states  (broadcast over headdim and d_state)
    # y_off[t] = sum_n C[t,n] * h_at_t[n,p] = sum_n C[t,n] * state_decay[t] * entering[n,p]
    y_off = torch.einsum('bcths,bchpn,bcshn->bcthp',
                          state_decay.permute(0, 1, 3, 2).unsqueeze(-1).expand(-1, -1, -1, -1, d_state).diagonal(dim1=-2, dim2=-1) if False else None,
                          entering_states, C)
    # That einsum is getting complicated. Let me be more explicit:

    # For each position t in chunk c:
    # y_off[b,c,t,h,p] = state_decay[b,c,h,t] * sum_n(C[b,c,t,h,n] * entering_states[b,c,h,p,n])

    # First: inner product of C and entering_states over d_state
    # C: (batch, nchunks, chunk_size, nheads, d_state)
    # entering: (batch, nchunks, nheads, headdim, d_state)
    # result: (batch, nchunks, chunk_size, nheads, headdim)
    y_state = torch.einsum('bcthn,bchpn->bcthp', C, entering_states)

    # Then multiply by decay
    # state_decay: (batch, nchunks, nheads, chunk_size)
    y_off = y_state * state_decay.permute(0, 1, 3, 2).unsqueeze(-1)  # (batch, nchunks, chunk_size, nheads, headdim)

    # ===== Combine =====
    y = y_diag + y_off

    # Add skip connection: D * x
    # D: (nheads,), x: (batch, nchunks, chunk_size, nheads, headdim)
    y = y + D[None, None, None, :, None] * x

    # Reshape back
    y = y.reshape(batch, L_padded, nheads, headdim)

    # Remove padding
    if pad > 0:
        y = y[:, :L]

    return y


def selective_scan(x, dt_A, B, C, D, chunk_size=64):
    """Drop-in API for Mamba-2 SSD scan.

    Args:
        x:    (batch, L, nheads, headdim)
        dt_A: (batch, L, nheads) — pre-multiplied dt*A (negative, log-space decay)
        B:    (batch, L, nheads, d_state)
        C:    (batch, L, nheads, d_state)
        D:    (nheads,)
        chunk_size: chunk size for parallel scan

    Returns:
        y: (batch, L, nheads, headdim)
    """
    return ssd_chunk_scan(x, dt_A, B, C, D, chunk_size=chunk_size)
