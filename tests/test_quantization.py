# tests/test_quantization.py
"""Tests for Int6 STE QAT and GPTQ-lite quantization."""
import torch
import pytest


def quantize_float_tensor_int6_ste(t: torch.Tensor, clip_range: int = 31) -> torch.Tensor:
    """STE quantization: forward uses quantized values, backward is identity.
    Scale is computed without gradient tracking so only the STE path carries grad."""
    with torch.no_grad():
        t32 = t.float()
        if t32.ndim == 2:
            row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
            q = torch.round(t32 / s[:, None]).clamp(-clip_range, clip_range)
            fake = s[:, None] * q
        else:
            amax = t32.abs().max()
            s = (amax / clip_range).clamp_min(1.0 / clip_range)
            q = torch.round(t32 / s).clamp(-clip_range, clip_range)
            fake = s * q
    return fake + t - t.detach()  # STE: forward=quantized, backward=identity


def gptq_lite_quantize_tensor(t: torch.Tensor, clip_range: int = 31):
    """Matches train_gpt.py implementation exactly."""
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.round(t32 / s.float()[:, None]).clamp(-clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    s = torch.tensor(max(amax / clip_range, 1.0 / clip_range), dtype=torch.float16)
    q = torch.round(t32 / s.float()).clamp(-clip_range, clip_range).to(torch.int8)
    return q, s


class TestInt6STE:
    def test_forward_values_in_range(self):
        """Quantized values must be in [-31, 31] range."""
        w = torch.randn(64, 64) * 2.0
        result = quantize_float_tensor_int6_ste(w)
        row_clip = w.float().abs().amax(dim=1)
        s = (row_clip / 31).clamp_min(1/31)
        q = (result.float() / s[:, None]).round()
        assert (q.abs() <= 31).all(), "Quantized values must be within [-31, 31]"

    def test_ste_gradient_is_identity(self):
        """STE: gradient through quantization should be identity (grad=1)."""
        w = torch.randn(8, 8, requires_grad=True)
        result = quantize_float_tensor_int6_ste(w)
        loss = result.sum()
        loss.backward()
        assert w.grad is not None
        assert torch.allclose(w.grad, torch.ones_like(w)), \
            "STE gradient must be identity (all 1s)"

    def test_quantization_reduces_unique_values(self):
        """After quantization, number of unique values should be <= 63 (int6 range)."""
        w = torch.randn(128, 128)
        result = quantize_float_tensor_int6_ste(w.detach())
        row_clip = w.abs().amax(dim=1)
        s = (row_clip / 31).clamp_min(1/31)
        q_int = (result.detach().float() / s[:, None]).round().int()
        assert q_int.unique().numel() <= 63, \
            "Int6 should produce at most 63 unique quantized values"

    def test_1d_tensor_quantization(self):
        """1D tensors (biases/scalars) should quantize without error."""
        b = torch.randn(512)
        result = quantize_float_tensor_int6_ste(b)
        assert result.shape == b.shape
        assert not torch.isnan(result).any()

    def test_zero_tensor_safe(self):
        """Zero tensor should not cause divide-by-zero."""
        w = torch.zeros(16, 16)
        result = quantize_float_tensor_int6_ste(w)
        assert not torch.isnan(result).any(), "Zero tensor must not produce NaN"
        assert not torch.isinf(result).any()


class TestGPTQLite:
    def test_picks_best_percentile(self):
        """GPTQ-lite should pick the percentile minimizing reconstruction MSE."""
        w = torch.randn(32, 32)
        w[0, 0] = 100.0  # outlier
        q, s = gptq_lite_quantize_tensor(w)
        recon = q.float() * s.float()[:, None]
        mse = (w.float() - recon).pow(2).mean().item()
        row_clip = w.float().abs().amax(dim=1)
        s_naive = (row_clip / 31).clamp_min(1/31).to(torch.float16)
        q_naive = torch.round(w.float() / s_naive.float()[:, None]).clamp(-31, 31).to(torch.int8)
        recon_naive = q_naive.float() * s_naive.float()[:, None]
        mse_naive = (w.float() - recon_naive).pow(2).mean().item()
        assert mse <= mse_naive + 1e-6, \
            "GPTQ-lite must achieve MSE <= naive amax quantization"

    def test_output_dtype_int8(self):
        """Output quantized tensor must be int8."""
        w = torch.randn(16, 16)
        q, s = gptq_lite_quantize_tensor(w)
        assert q.dtype == torch.int8, "Quantized output must be int8"

    def test_scale_dtype_float16(self):
        """Scale tensor must be float16."""
        w = torch.randn(16, 16)
        q, s = gptq_lite_quantize_tensor(w)
        assert s.dtype == torch.float16, "Scale must be float16"

    def test_reconstruction_roundtrip(self):
        """Reconstructed weights should be close to original (within quantization error)."""
        torch.manual_seed(42)
        w = torch.randn(64, 64)
        q, s = gptq_lite_quantize_tensor(w)
        recon = q.float() * s.float()[:, None]
        max_err = (w.float() - recon).abs().max().item()
        assert max_err < w.abs().max().item() * 0.1, \
            f"Reconstruction error too large: {max_err:.4f}"

    def test_1d_fallback(self):
        """1D tensors should use single-scale quantization."""
        v = torch.randn(256)
        q, s = gptq_lite_quantize_tensor(v)
        assert q.shape == v.shape
        assert s.numel() == 1, "1D tensor should produce a single scale factor"
