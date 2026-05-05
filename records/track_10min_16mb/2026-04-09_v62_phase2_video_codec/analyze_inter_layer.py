#!/usr/bin/env python3
"""Phase 2 sanity: analyze how similar weights are across layers in v6.1.

If `W[layer_i+1] - W[layer_i]` (the delta) has noticeably lower entropy
or smaller magnitude than W itself, then inter-layer delta prediction
will compress well via rANS. Otherwise the trick is dead.

Reads runs/v61_fa3_seq2048_s1337/model.pt (FP32 state_dict) and prints,
for every layer-N parameter that has a layer-(N-1) twin, the following:
  - W mean abs, W std
  - delta mean abs, delta std
  - delta magnitude ratio = delta_abs_mean / W_abs_mean
  - cosine similarity between flat W_i and W_{i-1}
  - if you Pentanary-quantize W vs delta, what is the symbol histogram
    entropy (in bits)?

Usage:
    python analyze_inter_layer.py runs/v61_fa3_seq2048_s1337/model.pt
"""
import sys
import math
import re
from collections import defaultdict

import numpy as np
import torch


def histogram_entropy_pent(t: torch.Tensor) -> float:
    """Pentanary symbol histogram entropy after PentanaryLinear quantization."""
    abs_t = t.abs()
    mean_abs = abs_t.mean(dim=1, keepdim=True)
    t1 = 0.7 * mean_abs
    t2 = 2.0 * t1
    mask1 = abs_t > t1
    mask2 = abs_t > t2
    q = torch.sign(t) * (mask1.float() + mask2.float())  # in {-2..+2}
    sym = (q + 2).long().flatten().numpy()
    counts = np.bincount(sym, minlength=5).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def histogram_entropy_int4(t: torch.Tensor) -> float:
    """Int4 (alphabet=16) per-row symbol entropy."""
    w_max = t.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    half = 8
    w_int = (t / w_max * half).round().clamp(-half, half - 1)
    sym = (w_int + half).long().flatten().numpy()
    counts = np.bincount(sym, minlength=16).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    pt = sys.argv[1]
    print(f"[analyze] loading {pt} ...")
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    if "model" in ckpt and "step" in ckpt:
        if "ema_shadow" in ckpt:
            ema = ckpt["ema_shadow"]
            sd = ema["smoother"] if "fast" in ema else ema
        else:
            sd = ckpt["model"]
    else:
        sd = ckpt

    # Group parameters by their template (e.g., "blocks.{}.attn.c_q.weight")
    pattern = re.compile(r"^blocks\.(\d+)\.(.+)$")
    by_template = defaultdict(dict)  # tmpl -> {layer_idx: tensor}
    for key, val in sd.items():
        m = pattern.match(key)
        if not m:
            continue
        if not isinstance(val, torch.Tensor):
            continue
        if val.ndim != 2 or val.shape[0] < 16 or val.shape[1] < 16:
            continue
        layer_idx = int(m.group(1))
        tmpl = m.group(2)
        by_template[tmpl][layer_idx] = val.float()

    print(f"\n[analyze] {len(by_template)} parameter templates, "
          f"{sum(len(v) for v in by_template.values())} total tensors")
    print()
    print(f"{'template':<35} {'shape':<15} {'W_abs':<10} {'d_abs':<10} {'ratio':<8} "
          f"{'H(W)pent':<10} {'H(d)pent':<10} {'H(W)int4':<10} {'H(d)int4':<10}")
    print("-" * 130)

    total_W_pent_bits = 0.0
    total_d_pent_bits = 0.0
    total_params = 0

    for tmpl, layers in sorted(by_template.items()):
        if len(layers) < 2:
            continue
        sorted_keys = sorted(layers.keys())
        first = sorted_keys[0]
        W0 = layers[first]
        for i in sorted_keys[1:]:
            W = layers[i]
            d = W - layers[i - 1] if (i - 1) in layers else (W - W0)
            w_abs = W.abs().mean().item()
            d_abs = d.abs().mean().item()
            ratio = d_abs / w_abs if w_abs > 0 else 0.0
            H_W_pent = histogram_entropy_pent(W)
            H_d_pent = histogram_entropy_pent(d)
            H_W_int4 = histogram_entropy_int4(W)
            H_d_int4 = histogram_entropy_int4(d)
            total_W_pent_bits += H_W_pent * W.numel()
            total_d_pent_bits += H_d_pent * d.numel()
            total_params += W.numel()
            print(f"{tmpl + '['+str(i)+']':<35} {str(tuple(W.shape)):<15} "
                  f"{w_abs:<10.5f} {d_abs:<10.5f} {ratio:<8.3f} "
                  f"{H_W_pent:<10.4f} {H_d_pent:<10.4f} {H_W_int4:<10.4f} {H_d_int4:<10.4f}")

    if total_params > 0:
        avg_W = total_W_pent_bits / total_params
        avg_d = total_d_pent_bits / total_params
        gain = avg_W - avg_d
        print()
        print(f"[summary] across {total_params:,} delta params (i>=1):")
        print(f"  pent H(W) avg = {avg_W:.4f} bits/sym")
        print(f"  pent H(delta) avg = {avg_d:.4f} bits/sym")
        print(f"  gain         = {gain:+.4f} bits/sym")
        if gain > 0:
            saved_bytes = gain * total_params / 8
            print(f"  potential savings (if pent + ideal entropy coding) = "
                  f"{saved_bytes:,.0f} bytes = {saved_bytes/2**20:.2f} MB")
        else:
            print("  → delta has HIGHER entropy than W, inter-layer prediction WORSE than direct.")


if __name__ == "__main__":
    main()
