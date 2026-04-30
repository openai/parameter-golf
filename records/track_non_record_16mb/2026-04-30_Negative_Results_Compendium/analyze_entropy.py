"""Analyze the distribution of quantized int values to understand what Brotli exploits.

Simulates PR-1493's SDClip quantization (without GPTQ compensation) and measures:
1. Histogram of quantized int values
2. Shannon entropy (theoretical minimum bits/value)
3. Zero/near-zero concentration
4. Per-tensor entropy variation
5. Comparison with tighter k

Usage:
    BUNDLE_DIR=./local_bundle_seed42 /tmp/torch_env/bin/python3 analyze_entropy.py
"""
import os, math, torch
from pathlib import Path
from collections import Counter

bundle_dir = Path(os.environ.get("BUNDLE_DIR", "local_bundle_seed42"))
ema = torch.load(bundle_dir / "ema_weights.pt", map_location="cpu")

def quantize_sdclip(w, k, clip_range):
    """Simulate SDClip quantization (no GPTQ compensation)."""
    row_std = w.std(dim=1, keepdim=True).clamp_min(1e-10)
    scale = k * row_std / clip_range
    q = torch.clamp(torch.round(w / scale), -clip_range, clip_range).to(torch.int8)
    return q

def shannon_entropy(q_flat):
    """Compute Shannon entropy in bits per value."""
    counts = Counter(q_flat.numpy().tolist())
    total = len(q_flat)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def analyze_config(label, k, clip_range, tensors):
    """Analyze quantized value distribution for a given k/clip_range."""
    all_q = []
    per_tensor_entropy = []

    for name, w in tensors:
        q = quantize_sdclip(w, k, clip_range)
        q_flat = q.flatten()
        all_q.append(q_flat)
        ent = shannon_entropy(q_flat)
        per_tensor_entropy.append((name, ent, q_flat.numel()))

    combined = torch.cat(all_q)
    total_entropy = shannon_entropy(combined)

    # Histogram
    counts = Counter(combined.numpy().tolist())
    total = len(combined)

    # Zero and near-zero stats
    zero_frac = counts.get(0, 0) / total
    near_zero = sum(counts.get(v, 0) for v in range(-2, 3)) / total  # [-2..2]
    mid_range = sum(counts.get(v, 0) for v in range(-7, 8)) / total  # [-7..7]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Shannon entropy:     {total_entropy:.3f} bits/value")
    print(f"  Theoretical min size: {total_entropy * total / 8 / 1e6:.1f} MB")
    print(f"  Values exactly 0:   {zero_frac:.1%}")
    print(f"  Values in [-2,+2]:  {near_zero:.1%}")
    print(f"  Values in [-7,+7]:  {mid_range:.1%}")
    print(f"  Unique values used: {len(counts)} / {2*clip_range+1}")

    # Top 10 most common values
    print(f"\n  Top 15 most common values:")
    for val, cnt in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        bar = '█' * int(cnt / total * 200)
        print(f"    q={val:>4d}: {cnt/total:>6.2%} {bar}")

    # Entropy range across tensors
    entropies = [e for _, e, _ in per_tensor_entropy]
    print(f"\n  Per-tensor entropy: min={min(entropies):.3f} max={max(entropies):.3f} "
          f"mean={sum(entropies)/len(entropies):.3f}")

    # Most and least entropic tensors
    per_tensor_entropy.sort(key=lambda x: x[1])
    print(f"  Lowest entropy:  {per_tensor_entropy[0][0]} ({per_tensor_entropy[0][1]:.3f} bits)")
    print(f"  Highest entropy: {per_tensor_entropy[-1][0]} ({per_tensor_entropy[-1][1]:.3f} bits)")

    return total_entropy

# Collect int6 tensors
int6_tensors = [(n, ema[n].float()) for n in sorted(ema)
                if n != "tok_emb.weight" and ema[n].numel() > 65536
                and ema[n].is_floating_point() and ema[n].ndim == 2]

print(f"Analyzing {len(int6_tensors)} tensors, {sum(w.numel() for _,w in int6_tensors):,} total elements")

# Analyze at different k values
e_baseline = analyze_config("Baseline: int6 k=12.85 (clip_range=31)", 12.85, 31, int6_tensors)
e_k6 = analyze_config("Tighter:  int6 k=6.0  (clip_range=31)", 6.0, 31, int6_tensors)
e_int5 = analyze_config("int5 k=6.0 (clip_range=15)", 6.0, 15, int6_tensors)
e_int5_wide = analyze_config("int5 k=12.85 (clip_range=15)", 12.85, 15, int6_tensors)

print(f"\n\n{'='*70}")
print(f"  SUMMARY: entropy vs Brotli")
print(f"{'='*70}")
print(f"  {'Config':<30} {'Entropy':>8} {'Theory min':>11} {'Actual Brotli':>14}")
print(f"  {'Baseline int6 k=12.85':<30} {e_baseline:>7.3f}b {e_baseline*31719424/8/1e6:>10.1f} MB {'15.97 MB':>14}")
print(f"  {'int6 k=6.0':<30} {e_k6:>7.3f}b {e_k6*31719424/8/1e6:>10.1f} MB {'20.23 MB':>14}")
print(f"  {'int5 k=6.0':<30} {e_int5:>7.3f}b {e_int5*31719424/8/1e6:>10.1f} MB {'16.18 MB':>14}")
print(f"  {'int5 k=12.85':<30} {e_int5_wide:>7.3f}b {e_int5_wide*31719424/8/1e6:>10.1f} MB {'(not tested)':>14}")
