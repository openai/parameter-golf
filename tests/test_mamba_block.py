"""Unit tests for MambaBlock — Story 2B.1

Tests:
  2B.1.1  Forward shape correctness
  2B.1.2  Gradient flow
  2B.1.3  Parameter count matches budget
  2B.1.4  SSM numerical correctness
  2B.1.5  Causal masking (no future leakage)
  2B.1.6  Determinism
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _fake_flash_attn(q, k, v, causal=False):
    """CPU fallback: naive scaled dot-product attention with GQA support."""
    bsz, seqlen, nqh, hd = q.shape
    nkvh = k.shape[2]
    if nqh != nkvh:
        reps = nqh // nkvh
        k = k.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
        v = v.unsqueeze(3).expand(bsz, seqlen, nkvh, reps, hd).reshape(bsz, seqlen, nqh, hd)
    scale = hd ** -0.5
    q2 = q.transpose(1, 2).float()
    k2 = k.transpose(1, 2).float()
    v2 = v.transpose(1, 2).float()
    attn = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v2)
    return out.transpose(1, 2).to(q.dtype)


# ---------------------------------------------------------------------------
# Mock unavailable GPU-only modules before importing train_gpt
# ---------------------------------------------------------------------------
for mod_name in ("flash_attn_interface", "flash_attn", "mamba_ssm", "causal_conv1d"):
    if mod_name not in sys.modules:
        fake = types.ModuleType(mod_name)
        fake.flash_attn_func = _fake_flash_attn
        sys.modules[mod_name] = fake
sys.modules["flash_attn_interface"].flash_attn_func = _fake_flash_attn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_gpt
train_gpt.flash_attn_3_func = _fake_flash_attn
from train_gpt import MambaBlock


# ===== 2B.1.1 — Forward shape correctness ================================

@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("seq_len", [1, 64, 128, 2048])
def test_forward_shape(batch_size: int, seq_len: int):
    """Input (B, L, D) -> output (B, L, D) for various B, L."""
    d_model = 512
    block = MambaBlock(d_model=d_model, d_state=32, d_conv=4, expand=1.5)
    block.eval()
    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (batch_size, seq_len, d_model), (
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {y.shape}"
    )


def test_forward_shape_single_token():
    """Edge case: single token (L=1)."""
    block = MambaBlock(d_model=512)
    block.eval()
    x = torch.randn(1, 1, 512)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 1, 512)


# ===== 2B.1.2 — Gradient flow ============================================

def test_gradient_flow():
    """Forward -> scalar loss -> .backward() -> all params have non-None, non-zero .grad."""
    block = MambaBlock(d_model=512, d_state=32, d_conv=4, expand=1.5)
    block.train()
    x = torch.randn(2, 64, 512, requires_grad=False)
    y = block(x)
    loss = y.sum()
    loss.backward()

    param_names_to_check = [
        "in_proj.weight",
        "out_proj.weight",
        "dt_proj.weight",
        "dt_proj.bias",
        "c_proj.weight",
        "A_log",
        "D",
        "conv1d.weight",
        "conv1d.bias",
    ]
    for name, param in block.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert param.grad.abs().sum() > 0, f"Gradient is all zeros for {name}"


def test_gradient_flow_all_named_params():
    """Every single named parameter must receive a gradient."""
    block = MambaBlock(d_model=512)
    x = torch.randn(2, 32, 512)
    y = block(x)
    loss = y.mean()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert torch.any(p.grad != 0), f"Zero grad for {name}"


# ===== 2B.1.3 — Parameter count matches budget ===========================

def _expected_param_count(d_model: int, d_state: int, d_conv: int, expand: float) -> int:
    """Compute expected parameter count for MambaBlock."""
    d_inner = int(expand * d_model)
    dt_rank = max(d_model // 16, 1)

    in_proj = d_model * (d_inner * 2 + d_state + dt_rank)  # no bias
    conv1d_w = d_inner * d_conv  # depthwise: groups=d_inner, so kernel is (d_inner, 1, d_conv)
    conv1d_b = d_inner
    dt_proj_w = dt_rank * d_inner
    dt_proj_b = d_inner
    a_log = d_inner * d_state
    d_param = d_inner
    c_proj = d_model * d_state  # no bias
    out_proj = d_inner * d_model  # no bias
    # norm has no parameters (RMSNorm without learnable weight in this impl)
    return in_proj + conv1d_w + conv1d_b + dt_proj_w + dt_proj_b + a_log + d_param + c_proj + out_proj


def test_param_count_default():
    """Default config (512, 32, 4, 1.5) should have ~1.27M params (±5%)."""
    block = MambaBlock(d_model=512, d_state=32, d_conv=4, expand=1.5)
    actual = sum(p.numel() for p in block.parameters())
    expected = _expected_param_count(512, 32, 4, 1.5)
    assert actual == expected, f"Param count mismatch: actual={actual}, expected={expected}"
    # Check ~1.27M (within ±5%)
    assert 1_207_000 <= actual <= 1_333_000, (
        f"Param count {actual} is outside expected range ~1.27M ±5%"
    )


@pytest.mark.parametrize("d_state", [16, 32, 64])
@pytest.mark.parametrize("expand", [1.0, 1.5, 2.0])
def test_param_count_configs(d_state: int, expand: float):
    """Param count matches budget for all ablation configs."""
    block = MambaBlock(d_model=512, d_state=d_state, d_conv=4, expand=expand)
    actual = sum(p.numel() for p in block.parameters())
    expected = _expected_param_count(512, d_state, 4, expand)
    assert actual == expected, (
        f"Param count mismatch for d_state={d_state}, expand={expand}: "
        f"actual={actual}, expected={expected}"
    )


# ===== 2B.1.4 — SSM numerical correctness ================================

def test_selective_scan_zero_input():
    """Zero input should produce output close to zero (only residual)."""
    block = MambaBlock(d_model=512, d_state=32)
    block.eval()
    x = torch.zeros(1, 16, 512)
    with torch.no_grad():
        y = block(x)
    # With zero input and residual connection: y = x + mamba_out = 0 + ~0
    assert y.abs().max() < 1.0, f"Output too large for zero input: max={y.abs().max()}"


def test_selective_scan_reference():
    """Compare _selective_scan output against manual computation for a simple case."""
    torch.manual_seed(42)
    block = MambaBlock(d_model=64, d_state=8, d_conv=4, expand=1.5)
    block.eval()
    d_inner = block.d_inner  # 96

    B_batch, L = 1, 4
    x = torch.randn(B_batch, L, d_inner)
    dA = torch.rand(B_batch, L, d_inner, 8) * 0.9  # keep < 1 for stability
    dB = torch.randn(B_batch, L, d_inner, 8) * 0.1
    C = torch.randn(B_batch, L, 8)
    D = torch.ones(d_inner)

    # Compute with method
    with torch.no_grad():
        y_method = block._selective_scan(x, dA, dB, C, D)

    # Manual reference
    h = torch.zeros(B_batch, d_inner, 8)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
        y_t = (h * C[:, t, None, :]).sum(-1) + D * x[:, t]
        ys.append(y_t)
    y_ref = torch.stack(ys, dim=1)

    assert torch.allclose(y_method, y_ref, atol=1e-5), (
        f"Selective scan output differs from reference: max_diff={( y_method - y_ref).abs().max()}"
    )


def test_selective_scan_identity_dynamics():
    """With dA=0 and D=0, output should only depend on current step (no memory)."""
    block = MambaBlock(d_model=64, d_state=8)
    d_inner = block.d_inner
    B_batch, L = 2, 8

    x = torch.randn(B_batch, L, d_inner)
    dA = torch.zeros(B_batch, L, d_inner, 8)  # no memory carryover
    dB = torch.ones(B_batch, L, d_inner, 8) * 0.1
    C = torch.ones(B_batch, L, 8) * 0.1
    D = torch.zeros(d_inner)

    with torch.no_grad():
        y = block._selective_scan(x, dA, dB, C, D)

    # With dA=0, h[t] = dB[t]*x[t] (no carryover), y[t] = C[t]@h[t]
    for t in range(L):
        h_t = dB[:, t] * x[:, t, :, None]  # (B, d_inner, d_state)
        y_expected = (h_t * C[:, t, None, :]).sum(-1)
        assert torch.allclose(y[:, t], y_expected, atol=1e-5), (
            f"Step {t} differs: max_diff={(y[:, t] - y_expected).abs().max()}"
        )


# ===== 2B.1.5 — Causal masking (no future leakage) =======================

def test_causal_no_future_leakage():
    """Output at position t should not depend on inputs at positions > t.
    Forward on length L, then on length L-1: outputs 0..L-2 must be identical.
    """
    torch.manual_seed(123)
    block = MambaBlock(d_model=512, d_state=32, d_conv=4, expand=1.5)
    block.eval()

    L = 32
    x_full = torch.randn(1, L, 512)
    x_trunc = x_full[:, :L - 1].clone()

    with torch.no_grad():
        y_full = block(x_full)
        y_trunc = block(x_trunc)

    # Outputs for positions 0..L-2 should be identical
    diff = (y_full[:, :L - 1] - y_trunc).abs().max()
    assert diff < 1e-5, f"Causal violation: outputs differ by {diff} for truncated input"


def test_causal_prefix_independence():
    """Changing a future token should not affect past outputs."""
    torch.manual_seed(456)
    block = MambaBlock(d_model=64, d_state=8, d_conv=4, expand=1.5)
    block.eval()

    L = 16
    x1 = torch.randn(1, L, 64)
    x2 = x1.clone()
    x2[:, L - 1, :] = torch.randn(64)  # change only last token

    with torch.no_grad():
        y1 = block(x1)
        y2 = block(x2)

    # All positions except the last should be identical
    diff = (y1[:, :L - 1] - y2[:, :L - 1]).abs().max()
    assert diff < 1e-5, f"Changing last token affected earlier outputs: diff={diff}"


# ===== 2B.1.6 — Determinism ==============================================

def test_determinism():
    """Same input + same seed -> same output across 3 runs (bitwise identical in fp32)."""
    d_model = 512
    outputs = []
    for _ in range(3):
        torch.manual_seed(999)
        block = MambaBlock(d_model=d_model, d_state=32, d_conv=4, expand=1.5)
        block.eval()
        x = torch.randn(2, 64, d_model)
        with torch.no_grad():
            y = block(x)
        outputs.append(y)

    assert torch.equal(outputs[0], outputs[1]), "Run 1 != Run 2"
    assert torch.equal(outputs[1], outputs[2]), "Run 2 != Run 3"


def test_determinism_train_mode():
    """Determinism holds in train mode too (no dropout in MambaBlock)."""
    outputs = []
    for _ in range(3):
        torch.manual_seed(42)
        block = MambaBlock(d_model=64, d_state=8)
        block.train()
        x = torch.randn(1, 16, 64)
        y = block(x)
        outputs.append(y.detach().clone())

    assert torch.equal(outputs[0], outputs[1]), "Train mode: Run 1 != Run 2"
    assert torch.equal(outputs[1], outputs[2]), "Train mode: Run 2 != Run 3"


# ===== Additional edge cases =============================================

def test_output_finite():
    """Forward pass should never produce NaN or Inf."""
    torch.manual_seed(7)
    block = MambaBlock(d_model=512, d_state=32, d_conv=4, expand=1.5)
    block.eval()
    x = torch.randn(4, 128, 512)
    with torch.no_grad():
        y = block(x)
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"


def test_residual_connection():
    """At initialization with small out_proj, output should be close to input (residual dominant)."""
    torch.manual_seed(0)
    block = MambaBlock(d_model=512, d_state=32, d_conv=4, expand=1.5)
    block.eval()
    x = torch.randn(1, 32, 512)
    with torch.no_grad():
        y = block(x)
    # Residual connection: y ≈ x + small_correction
    rel_diff = (y - x).norm() / x.norm()
    assert rel_diff < 5.0, f"Output too far from residual: relative diff = {rel_diff}"
