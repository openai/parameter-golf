#!/usr/bin/env python3
"""
SDClip k-sweep: Re-quantize a trained model at different clipping thresholds.
No retraining needed. Measures post-quantization BPB for each k value.

Usage:
    python3 k_sweep.py --model final_model.pt

This finds the optimal SDClip k for YOUR specific model's weight distribution.
"""

import argparse
import io
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def compute_weight_stats(state_dict):
    """Analyze weight distributions to understand what k values make sense."""
    print("\n=== Weight Distribution Analysis ===\n")

    total_params = 0
    total_outliers = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    max_sigma = 0

    for name, w in state_dict.items():
        if w.dim() < 2 or w.numel() < 100:
            continue

        w_flat = w.float().flatten()
        std = w_flat.std().item()
        if std == 0:
            continue

        normalized = (w_flat / std).abs()
        max_val = normalized.max().item()
        max_sigma = max(max_sigma, max_val)
        n = w_flat.numel()
        total_params += n

        for k in total_outliers:
            total_outliers[k] += (normalized > k).sum().item()

    print(f"Total weight parameters: {total_params:,}")
    print(f"Maximum |weight|/std: {max_sigma:.1f}σ")
    print()
    print("Outlier counts:")
    for k in sorted(total_outliers):
        count = total_outliers[k]
        pct = count / total_params * 100 if total_params > 0 else 0
        print(f"  Beyond {k}σ: {count:>8,} ({pct:.6f}%)")
    print()

    # Kurtosis (tail heaviness)
    all_weights = []
    for name, w in state_dict.items():
        if w.dim() >= 2 and w.numel() >= 100:
            all_weights.append(w.float().flatten())

    if all_weights:
        combined = torch.cat(all_weights)
        mean = combined.mean()
        std = combined.std()
        kurtosis = ((combined - mean) / std).pow(4).mean().item()
        print(f"Kurtosis: {kurtosis:.2f} (Gaussian=3.0, higher=heavier tails)")
        if kurtosis > 4:
            print(f"  → Heavy tails detected. Optimal k likely 7-9.")
        elif kurtosis > 3.5:
            print(f"  → Moderate tails. Optimal k likely 6-8.")
        else:
            print(f"  → Near-Gaussian. Optimal k likely 5-7.")


def quantize_tensor_int6(w, k):
    """Quantize a weight tensor to int6 with SDClip at k sigma."""
    w_float = w.float()

    if w_float.dim() < 2:
        # Don't quantize 1D tensors (biases, scales)
        return w_float, 0.0

    # Per-row quantization
    std = w_float.std(dim=-1, keepdim=True)
    clip_val = k * std

    # Clip
    w_clipped = w_float.clamp(-clip_val, clip_val)

    # Count clipped values
    n_clipped = ((w_float.abs() > clip_val).sum()).item()

    # Quantize to 64 levels
    w_min = w_clipped.min(dim=-1, keepdim=True).values
    w_max = w_clipped.max(dim=-1, keepdim=True).values
    w_range = w_max - w_min
    w_range = torch.where(w_range == 0, torch.ones_like(w_range), w_range)

    # Quantize
    w_norm = (w_clipped - w_min) / w_range
    w_quant = (w_norm * 63).round().clamp(0, 63)

    # Dequantize
    w_deq = w_quant / 63 * w_range + w_min

    # Compute MSE
    mse = ((w_float - w_deq) ** 2).mean().item()

    return w_deq, mse


def simulate_quantized_loss(state_dict, k, device='cpu'):
    """Quantize all weights at given k and return total MSE."""
    total_mse = 0.0
    total_params = 0
    total_clipped = 0

    for name, w in state_dict.items():
        if w.dim() < 2 or w.numel() < 100:
            continue

        w_float = w.float()
        std = w_float.std(dim=-1, keepdim=True)
        clip_val = k * std
        n_clipped = ((w_float.abs() > clip_val).sum()).item()
        total_clipped += n_clipped

        _, mse = quantize_tensor_int6(w, k)
        total_mse += mse * w.numel()
        total_params += w.numel()

    avg_mse = total_mse / total_params if total_params > 0 else 0
    return avg_mse, total_clipped, total_params


def main():
    parser = argparse.ArgumentParser(description="SDClip k-sweep")
    parser.add_argument("--model", required=True, help="Path to trained model .pt file")
    parser.add_argument("--k-values", default="4,5,6,7,8,9,10,11,12,13", help="Comma-separated k values to test")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    state_dict = torch.load(args.model, map_location=args.device, weights_only=True)
    print(f"Loaded {len(state_dict)} tensors")

    # Analyze weight distribution
    compute_weight_stats(state_dict)

    # Sweep k values
    k_values = [float(k) for k in args.k_values.split(",")]

    print("=== SDClip k-sweep ===\n")
    print(f"{'k':>6} {'Avg MSE':>12} {'Clipped':>10} {'Clipped%':>10} {'Relative MSE':>14}")
    print("-" * 60)

    results = []
    baseline_mse = None

    for k in k_values:
        t0 = time.time()
        avg_mse, n_clipped, n_params = simulate_quantized_loss(state_dict, k, args.device)
        elapsed = time.time() - t0

        if baseline_mse is None:
            baseline_mse = avg_mse

        relative = avg_mse / baseline_mse if baseline_mse > 0 else 1.0
        clipped_pct = n_clipped / n_params * 100 if n_params > 0 else 0

        results.append((k, avg_mse, n_clipped, clipped_pct, relative))
        print(f"{k:>6.1f} {avg_mse:>12.8f} {n_clipped:>10,} {clipped_pct:>9.4f}% {relative:>13.3f}x")

    # Find optimal k
    best_k, best_mse = min(results, key=lambda x: x[1])[:2]

    print(f"\n{'='*60}")
    print(f"Optimal k = {best_k}")
    print(f"MSE at optimal: {best_mse:.8f}")
    print(f"MSE at k=12.85: {results[-1][1]:.8f}" if 12.85 in k_values or 13 in k_values else "")

    improvement = (1 - best_mse / results[-1][1]) * 100 if len(results) > 1 else 0
    print(f"MSE improvement vs highest k: {improvement:.1f}%")

    print(f"\nRecommendation: Use SDCLIP_SIGMA={best_k} in your training config")
    print(f"Expected BPB gain: ~{improvement/100 * 0.024:.4f} BPB (rough estimate)")


if __name__ == "__main__":
    main()
