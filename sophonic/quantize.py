"""
Sophonic-aware quantization utilities.

Extends the competition's int6 QAT with int5 base + residual correction.
"""
import torch


def int_quantize_per_row(x: torch.Tensor, bits: int, clip_percentile: float = 1.0):
    """Per-row symmetric integer quantization."""
    qmax = (1 << (bits - 1)) - 1
    if clip_percentile < 1.0:
        row_max = x.abs().quantile(clip_percentile, dim=-1, keepdim=True)
    else:
        row_max = x.abs().amax(dim=-1, keepdim=True)
    row_max = row_max.clamp(min=1e-8)
    scale = row_max / qmax
    x_int = (x / scale).round().clamp(-qmax, qmax)
    return x_int.to(torch.int8), scale


def sophonic_dequantize(x_int5: torch.Tensor, scale: torch.Tensor,
                         gate: float,
                         u_r: torch.Tensor = None, v_r: torch.Tensor = None):
    """
    Dequantize with optional low-rank precision correction.
    
    W_eff = dequant_int5(x_int5, scale) + gate * (U_r @ V_r.T)
    """
    w = x_int5.float() * scale
    if gate > 0 and u_r is not None and v_r is not None:
        w = w + gate * (u_r @ v_r.T).to(w.dtype)
    return w
