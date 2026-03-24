"""
Low-rank precision residuals.

Computes and stores the difference between high-precision and low-precision
quantized weights as rank-R approximations.
"""
import torch
from typing import Dict, Tuple, List


def compute_residual(weight_fp: torch.Tensor, q_low_bits: int, q_high_bits: int):
    """Compute the quantization residual: Q_high(W) - Q_low(W)."""
    q_low = fake_quantize(weight_fp, q_low_bits)
    q_high = fake_quantize(weight_fp, q_high_bits)
    return q_high - q_low


def low_rank_approximate(residual: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate residual with rank-R SVD.
    Returns (U_r, V_r) where residual ≈ U_r @ V_r.T
    U_r: (m, rank), V_r: (n, rank) — absorb singular values into U.
    """
    U, S, Vh = torch.linalg.svd(residual.float(), full_matrices=False)
    U_r = U[:, :rank] * S[:rank].unsqueeze(0)  # (m, rank) — absorb S into U
    V_r = Vh[:rank, :].T                        # (n, rank)
    return U_r.half(), V_r.half()


def fake_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row fake quantization with STE."""
    qmax = (1 << (bits - 1)) - 1
    row_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = row_max / qmax
    x_q = (x / scale).round().clamp(-qmax, qmax) * scale
    return x_q


def reconstruction_error(weight_fp: torch.Tensor, q_low_bits: int,
                          rank: int, q_high_bits: int = 8) -> Dict[str, float]:
    """Measure how well rank-R residuals capture the quantization gap."""
    residual = compute_residual(weight_fp, q_low_bits, q_high_bits)
    U_r, V_r = low_rank_approximate(residual, rank)
    reconstructed = U_r @ V_r.T
    
    total_energy = (residual ** 2).sum().item()
    error_energy = ((residual - reconstructed.to(residual.dtype)) ** 2).sum().item()
    
    return {
        "total_residual_norm": total_energy ** 0.5,
        "reconstruction_error": error_energy ** 0.5,
        "energy_captured_pct": 100 * (1 - error_energy / max(total_energy, 1e-12)),
        "rank": rank,
    }
