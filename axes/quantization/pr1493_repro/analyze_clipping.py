"""Analyze weight clipping rates at various k values for SDClip.

Loads EMA weights from bundle and reports what fraction of values
get clipped at each k level. No GPU needed — pure CPU tensor math.

Usage:
    BUNDLE_DIR=./bundle python3 analyze_clipping.py
"""
import os, sys, torch
from pathlib import Path

bundle_dir = Path(os.environ.get("BUNDLE_DIR", "bundle"))
ema = torch.load(bundle_dir / "ema_weights.pt", map_location="cpu")

# Which tensors get int6 in PR-1493 (the "else" branch: not tok_emb, not small)
int6_names = [n for n in ema if n != "tok_emb.weight" and ema[n].numel() > 65536 and ema[n].is_floating_point()]
int8_names = [n for n in ema if "tok_emb" in n and ema[n].numel() > 65536]

k_values = [3.0, 6.0, 9.0, 10.0, 11.0, 12.0, 12.85, 15.0, 20.0]

print(f"{'tensor':<45} {'shape':>14} {'row_std':>8}", end="")
for k in k_values:
    print(f"  k={k:<5}", end="")
print()
print("-" * (45 + 14 + 8 + len(k_values) * 10))

total_elements = 0
total_clipped = {k: 0 for k in k_values}

for name in sorted(int6_names):
    w = ema[name].float()
    if w.ndim != 2:
        continue
    rows, cols = w.shape
    row_std = w.std(dim=1)  # (rows,)
    avg_std = row_std.mean().item()

    print(f"{name:<45} {str(list(w.shape)):>14} {avg_std:>8.5f}", end="")
    for k in k_values:
        clip_boundary = k * row_std  # (rows,)
        clipped = (w.abs() > clip_boundary.unsqueeze(1)).sum().item()
        frac = clipped / w.numel()
        total_clipped[k] += clipped
        print(f"  {frac:>7.4%}", end="")
    print()
    total_elements += w.numel()

# tok_emb (int8, k=20)
for name in sorted(int8_names):
    w = ema[name].float()
    if w.ndim != 2:
        continue
    row_std = w.std(dim=1)
    avg_std = row_std.mean().item()
    print(f"{name:<45} {str(list(w.shape)):>14} {avg_std:>8.5f}", end="")
    for k in k_values:
        clip_boundary = k * row_std
        clipped = (w.abs() > clip_boundary.unsqueeze(1)).sum().item()
        frac = clipped / w.numel()
        print(f"  {frac:>7.4%}", end="")
    print(f"  ← int8 k=20")

print()
print(f"{'AGGREGATE (int6 tensors)':<45} {'':>14} {'':>8}", end="")
for k in k_values:
    frac = total_clipped[k] / total_elements if total_elements > 0 else 0
    print(f"  {frac:>7.4%}", end="")
print(f"\n\nTotal int6 elements: {total_elements:,}")
print(f"At k=12.85: {total_clipped[12.85]:,} clipped ({total_clipped[12.85]/total_elements:.6%})")
print(f"At k=6.0:   {total_clipped[6.0]:,} clipped ({total_clipped[6.0]/total_elements:.6%})")
print(f"At k=3.0:   {total_clipped[3.0]:,} clipped ({total_clipped[3.0]/total_elements:.6%})")

# Distribution shape analysis
print(f"\n{'tensor':<45} {'kurtosis':>10} {'skewness':>10}  shape")
print("-" * 80)
for name in sorted(int6_names)[:10]:
    w = ema[name].float().flatten()
    mean = w.mean()
    std = w.std()
    z = (w - mean) / std
    kurt = z.pow(4).mean().item() - 3.0  # excess kurtosis (0 = Gaussian)
    skew = z.pow(3).mean().item()
    shape = "Gaussian" if abs(kurt) < 1 else ("heavy-tail" if kurt > 1 else "light-tail")
    print(f"{name:<45} {kurt:>10.3f} {skew:>10.3f}  {shape}")
