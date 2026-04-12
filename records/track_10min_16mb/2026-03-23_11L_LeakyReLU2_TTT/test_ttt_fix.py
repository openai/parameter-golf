#!/usr/bin/env python3
"""
Smoke test for TTT fix: verifies Rotary cache doesn't break autograd
after scoring under inference_mode().

Run locally:  python3 records/track_10min_16mb/2026-03-23_11L_LeakyReLU2_TTT/test_ttt_fix.py
"""
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Minimal Rotary (same logic as in train_gpt.py) ──────────────────────────

class Rotary(nn.Module):
    def __init__(self, dim=32, rope_dims=16):
        super().__init__()
        self.rope_dims = rope_dims
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rope, x_pass), dim=-1)

# ── Mini-model that uses Rotary ──────────────────────────────────────────────

class TinyAttn(nn.Module):
    def __init__(self, dim=64, heads=4, rope_dims=16):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_dims = rope_dims
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(dim=self.head_dim, rope_dims=rope_dims)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.q(x).reshape(B, T, self.heads, self.head_dim)
        k = self.k(x).reshape(B, T, self.heads, self.head_dim)
        v = self.v(x).reshape(B, T, self.heads, self.head_dim)
        cos, sin = self.rotary(T, x.device, x.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        # Simple attention (not causal, just for test)
        q2 = q.transpose(1, 2)
        k2 = k.transpose(1, 2)
        v2 = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q2, k2, v2)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttn()
        self.fc = nn.Linear(64, 64)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(F.relu(self.attn(x)))

    def forward_logits(self, x: Tensor) -> Tensor:
        return self.forward(x)

# ── Test ─────────────────────────────────────────────────────────────────────

def test_without_fix(model):
    """Reproduces the original bug: should raise RuntimeError."""
    print("  [without fix] scoring under inference_mode...")
    with torch.inference_mode():
        x = torch.randn(2, 32, 64)
        _ = model.forward_logits(x)

    print("  [without fix] switching to training and calling backward...")
    model.train()
    # NO cache clear here — this is the BUG
    x = torch.randn(2, 32, 64)
    try:
        out = model(x)
        loss = out.sum()
        loss.backward()
        print("  [without fix] No error (PyTorch version may handle it differently)")
        return False  # No error = bug might be fixed in newer PyTorch
    except RuntimeError as e:
        if "Inference tensors cannot be saved for backward" in str(e):
            print(f"  [without fix] Got expected error: {e}")
            return True
        raise

def test_with_fix(model):
    """Tests the fix: should NOT raise RuntimeError."""
    print("  [with fix] scoring under inference_mode...")
    with torch.inference_mode():
        x = torch.randn(2, 32, 64)
        _ = model.forward_logits(x)

    model.train()
    # FIX: clear Rotary cache before training
    for m in model.modules():
        if hasattr(m, '_seq_len_cached'):
            m._seq_len_cached = 0
    print("  [with fix] Rotary cache cleared")

    print("  [with fix] calling backward...")
    x = torch.randn(2, 32, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()
    print(f"  [with fix] backward OK, grad norm: {x.grad.norm().item() if x.requires_grad else 'N/A'}")
    return True

def main():
    print("=" * 60)
    print("TTT Fix Smoke Test")
    print("=" * 60)

    torch.manual_seed(42)

    print("\n[1/2] Testing without fix (should show the bug or skip if PyTorch handles it):")
    model1 = TinyModel()
    bug_reproduced = test_without_fix(model1)

    print("\n[2/2] Testing with fix (should succeed):")
    model2 = TinyModel()
    try:
        test_with_fix(model2)
        print("\n✓ PASS — TTT fix works correctly")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ FAIL — Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
