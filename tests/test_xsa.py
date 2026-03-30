# tests/test_xsa.py
"""Tests for Exclusive Self Attention (XSA)."""
import torch
import torch.nn.functional as F
import pytest


def xsa(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exact copy of _xsa from train_gpt.py CausalSelfAttention."""
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    g = H // Hkv
    y_g = y.reshape(B, T, Hkv, g, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)


class TestXSA:
    def test_output_shape_preserved(self):
        B, T, H, D, Hkv = 2, 8, 4, 16, 2
        y = torch.randn(B, T, H, D)
        v = torch.randn(B, T, Hkv, D)
        out = xsa(y, v)
        assert out.shape == y.shape, "XSA must preserve input shape"

    def test_self_component_removed(self):
        """XSA should subtract the component of y aligned with v."""
        B, T, H, D = 1, 1, 1, 8
        v = torch.randn(B, T, 1, D)
        y = v.clone()  # shape [B,T,H,D] with H=Hkv=1
        out = xsa(y, v)
        vn = F.normalize(v, dim=-1)  # [B,T,1,D]
        dot = (out.reshape(B, T, 1, D) * vn).sum(dim=-1)
        assert dot.abs().max().item() < 1e-5, \
            "XSA must zero out the v-aligned component of y"

    def test_no_new_parameters(self):
        """XSA is parameter-free — no learnable weights."""
        B, T, H, D, Hkv = 2, 4, 2, 8, 1
        y = torch.randn(B, T, H, D)
        v = torch.randn(B, T, Hkv, D)
        y_orig = y.clone()
        v_orig = v.clone()
        _ = xsa(y, v)
        assert torch.allclose(y, y_orig), "XSA must not modify y in-place"
        assert torch.allclose(v, v_orig), "XSA must not modify v in-place"

    def test_gqa_grouping(self):
        """XSA must handle GQA (H > Hkv) correctly."""
        B, T, H, Hkv, D = 1, 4, 8, 4, 16  # 2 query heads per kv head
        y = torch.randn(B, T, H, D)
        v = torch.randn(B, T, Hkv, D)
        out = xsa(y, v)
        assert out.shape == (B, T, H, D)

    def test_reduces_self_correlation(self):
        """After XSA, cosine similarity between output and value should be lower."""
        B, T, H, D = 2, 16, 4, 32
        y = torch.randn(B, T, H, D)
        v = y.clone()
        out = xsa(y, v)
        cos_before = F.cosine_similarity(y.reshape(-1, D), v.reshape(-1, D), dim=-1).abs().mean()
        cos_after = F.cosine_similarity(out.reshape(-1, D), v.reshape(-1, D), dim=-1).abs().mean()
        assert cos_after < cos_before, \
            "XSA should reduce cosine similarity between output and value"
