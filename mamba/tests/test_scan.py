"""Test chunked selective scan against a naive sequential implementation."""

import torch
import pytest

from ..model import selective_scan


def naive_scan(x, A, B, C, dt):
    """Naive sequential scan — no chunking, pure loop over time."""
    B_batch, L, n_heads, head_dim = x.shape
    d_state = B.shape[-1]
    decay = torch.exp(A * dt)

    h = torch.zeros(B_batch, n_heads, head_dim, d_state, device=x.device, dtype=x.dtype)
    y = torch.zeros_like(x)

    for t in range(L):
        x_t = x[:, t]   # [B, n_heads, head_dim]
        B_t = B[:, t]    # [B, n_heads, d_state]
        C_t = C[:, t]    # [B, n_heads, d_state]
        h = decay[None, :, None, None] * h + torch.einsum("bhs,bhd->bhds", B_t, x_t) * dt[None, :, None, None]
        y_t = torch.einsum("bhs,bhds->bhd", C_t, h)
        y[:, t] = y_t

    return y


@pytest.mark.parametrize("chunk_size", [8, 16, 64])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_chunked_scan_matches_naive(chunk_size, seq_len):
    torch.manual_seed(42)
    B_batch, n_heads, head_dim, d_state = 2, 4, 32, 8

    x = torch.randn(B_batch, seq_len, n_heads, head_dim)
    A = -torch.abs(torch.randn(n_heads))  # negative
    B_proj = torch.randn(B_batch, seq_len, n_heads, d_state)
    C_proj = torch.randn(B_batch, seq_len, n_heads, d_state)
    dt = torch.abs(torch.randn(n_heads)) * 0.1

    y_chunked = selective_scan(x, A, B_proj, C_proj, dt, chunk_size)
    y_naive = naive_scan(x, A, B_proj, C_proj, dt)

    torch.testing.assert_close(y_chunked, y_naive, atol=1e-5, rtol=1e-4)


def test_scan_single_step():
    """Single timestep should match manual computation."""
    torch.manual_seed(0)
    B_batch, n_heads, head_dim, d_state = 1, 2, 4, 3

    x = torch.randn(B_batch, 1, n_heads, head_dim)
    A = -torch.ones(n_heads)
    B_proj = torch.randn(B_batch, 1, n_heads, d_state)
    C_proj = torch.randn(B_batch, 1, n_heads, d_state)
    dt = torch.ones(n_heads) * 0.1

    y = selective_scan(x, A, B_proj, C_proj, dt, chunk_size=1)

    # Manual: h = 0*decay + B outer x * dt; y = C . h
    decay = torch.exp(A * dt)
    h = torch.einsum("bhs,bhd->bhds", B_proj[:, 0], x[:, 0]) * dt[None, :, None, None]
    y_expected = torch.einsum("bhs,bhds->bhd", C_proj[:, 0], h)

    torch.testing.assert_close(y[:, 0], y_expected, atol=1e-6, rtol=1e-5)


def test_scan_zero_input():
    """Zero input should produce zero output."""
    B_batch, seq_len, n_heads, head_dim, d_state = 2, 16, 4, 8, 4
    x = torch.zeros(B_batch, seq_len, n_heads, head_dim)
    A = -torch.ones(n_heads)
    B_proj = torch.randn(B_batch, seq_len, n_heads, d_state)
    C_proj = torch.randn(B_batch, seq_len, n_heads, d_state)
    dt = torch.ones(n_heads) * 0.05

    y = selective_scan(x, A, B_proj, C_proj, dt, chunk_size=8)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)
