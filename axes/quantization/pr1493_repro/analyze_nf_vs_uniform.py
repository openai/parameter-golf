"""Compare uniform vs NormalFloat quantization MSE on saved weights.

NF (NormalFloat) places bins at Gaussian quantiles — more bins near zero
where density is highest. If weights are Gaussian, NF is information-
theoretically optimal. This script measures how much MSE improvement NF
gives over uniform at each bit width, per tensor.

Usage:
    BUNDLE_DIR=./local_bundle_seed42 /tmp/torch_env/bin/python3 analyze_nf_vs_uniform.py
"""
import os, math, torch
from pathlib import Path

bundle_dir = Path(os.environ.get("BUNDLE_DIR", "local_bundle_seed42"))
ema = torch.load(bundle_dir / "ema_weights.pt", map_location="cpu")

# Build NF quantization levels for a standard normal distribution.
# These are the conditional expectations of 2^b equal-probability bins.
def nf_levels(bits):
    n = 2 ** bits
    # Bin boundaries: quantiles at 0/n, 1/n, ..., n/n
    boundaries = torch.erfinv(2 * torch.linspace(0, 1, n + 1) - 1) * math.sqrt(2)
    boundaries[0] = -10.0   # -inf
    boundaries[-1] = 10.0   # +inf
    # Centroids: E[X | boundary_i < X < boundary_{i+1}] for standard normal
    # = (phi(b_i) - phi(b_{i+1})) / (Phi(b_{i+1}) - Phi(b_i))
    # where phi = PDF, Phi = CDF
    phi = lambda x: torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    levels = (phi(boundaries[:-1]) - phi(boundaries[1:])) / (1.0 / n)
    return levels  # (2^bits,) tensor of reconstruction levels in z-score space

def quantize_uniform(z, k, clip_range):
    """Uniform quantization in z-score space. Returns reconstruction and MSE."""
    scale = k / clip_range
    q = torch.clamp(torch.round(z / scale), -clip_range, clip_range)
    recon = q * scale
    mse = (z - recon).pow(2).mean().item()
    return mse

def quantize_nf(z, levels):
    """NF quantization: snap each value to nearest level. Returns MSE."""
    # z: (rows, cols), levels: (n_levels,)
    z_flat = z.reshape(-1, 1)  # (N, 1)
    dists = (z_flat - levels.unsqueeze(0)).abs()  # (N, n_levels)
    nearest = levels[dists.argmin(dim=1)]  # (N,)
    mse = (z_flat.squeeze() - nearest).pow(2).mean().item()
    return mse

# Precompute NF levels for each bit width
bit_widths = [4, 5, 6, 7, 8]
nf_cache = {b: nf_levels(b) for b in bit_widths}

# Identify int6 tensors (same rule as PR-1493)
int6_names = sorted([
    n for n in ema
    if n != "tok_emb.weight" and ema[n].numel() > 65536 and ema[n].is_floating_point() and ema[n].ndim == 2
])

print(f"Comparing uniform vs NF quantization MSE across {len(int6_names)} tensors")
print(f"All values normalized per-row (z = w / row_std)")
print()

# Header
print(f"{'bits':>4}  {'uniform k=12.85':>16}  {'uniform k=6':>14}  {'uniform k=3':>14}  {'NF (optimal)':>14}  {'NF/uni12.85':>12}  {'NF/uni6':>10}")
print("-" * 100)

# Aggregate across all tensors
for bits in bit_widths:
    clip_range = 2 ** (bits - 1) - 1  # 7 for int4, 15 for int5, 31 for int6, etc.

    total_mse_uni_1285 = 0.0
    total_mse_uni_6 = 0.0
    total_mse_uni_3 = 0.0
    total_mse_nf = 0.0
    total_elements = 0

    for name in int6_names:
        w = ema[name].float()
        row_std = w.std(dim=1, keepdim=True).clamp_min(1e-10)
        z = w / row_std  # normalize to z-scores per row
        n = w.numel()

        total_mse_uni_1285 += quantize_uniform(z, k=12.85, clip_range=clip_range) * n
        total_mse_uni_6 += quantize_uniform(z, k=6.0, clip_range=clip_range) * n
        total_mse_uni_3 += quantize_uniform(z, k=3.0, clip_range=clip_range) * n
        total_mse_nf += quantize_nf(z, nf_cache[bits]) * n
        total_elements += n

    mse_u1285 = total_mse_uni_1285 / total_elements
    mse_u6 = total_mse_uni_6 / total_elements
    mse_u3 = total_mse_uni_3 / total_elements
    mse_nf = total_mse_nf / total_elements

    ratio_vs_1285 = mse_nf / mse_u1285
    ratio_vs_6 = mse_nf / mse_u6

    print(f"{bits:>4}  {mse_u1285:>16.8f}  {mse_u6:>14.8f}  {mse_u3:>14.8f}  {mse_nf:>14.8f}  {ratio_vs_1285:>11.3f}x  {ratio_vs_6:>9.3f}x")

print()
print("Interpretation:")
print("  NF/uni ratio < 1 means NF is better (lower MSE).")
print("  NF/uni12.85 shows gain vs current PR-1493 setting.")
print("  NF/uni6 shows gain vs a tighter-but-still-uniform approach.")

# Per-tensor breakdown at 6 bits (the current setting)
print(f"\n\n--- Per-tensor breakdown at 6 bits ---")
print(f"{'tensor':<45} {'uni k=12.85':>12} {'uni k=6':>12} {'NF6':>12} {'NF6/uni12.85':>13} {'kurtosis':>9}")
print("-" * 105)

bits = 6
clip_range = 31
for name in int6_names:
    w = ema[name].float()
    row_std = w.std(dim=1, keepdim=True).clamp_min(1e-10)
    z = w / row_std

    mse_u1285 = quantize_uniform(z, k=12.85, clip_range=clip_range)
    mse_u6 = quantize_uniform(z, k=6.0, clip_range=clip_range)
    mse_nf = quantize_nf(z, nf_cache[6])

    # Kurtosis
    z_flat = z.flatten()
    kurt = z_flat.pow(4).mean().item() - 3.0

    ratio = mse_nf / mse_u1285
    print(f"{name:<45} {mse_u1285:>12.8f} {mse_u6:>12.8f} {mse_nf:>12.8f} {ratio:>12.3f}x {kurt:>9.3f}")
