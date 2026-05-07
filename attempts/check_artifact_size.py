"""
Artifact Size Checker for Parameter Golf

Checks whether the hypernetwork weights + code fit within the 16MB artifact limit.
We only need to ship:
  1. The hypernetwork weights (quantized + compressed)
  2. The code (hypernetwork.py + enough of train_gpt to define the target GPT)

The target GPT weights are generated at load time — they are NOT in the artifact.

Usage:
    python check_artifact_size.py [--budget BYTES] [--sweep]
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import zlib
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Add hypernetwork subdir to path so we can import
sys.path.insert(0, str(Path(__file__).parent / "hypernetwork"))
from hypernetwork import HyperNetwork, HyperNetConfig, TargetGPTConfig


# ---------------------------------------------------------------------------
# Quantization (mirrors baseline int8 approach)
# ---------------------------------------------------------------------------

def quantize_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row int8 quantization for 2D, per-tensor for others."""
    t32 = t.float()
    if t32.ndim == 2:
        amax = t32.abs().amax(dim=1).clamp_min(1e-8)
        scale = (amax / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    else:
        amax = float(t32.abs().max().item())
        scale = torch.tensor(max(amax / 127.0, 1.0 / 127.0), dtype=torch.float32)
        q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8)
        return q.contiguous(), scale


def quantize_state_dict(state_dict: dict[str, Tensor]) -> dict:
    """Quantize a state dict for size estimation."""
    quantized = {}
    scales = {}
    passthrough = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        # Small tensors: keep as fp16
        if t.numel() <= 65_536:
            if t.is_floating_point():
                passthrough[name] = t.to(torch.float16).contiguous()
            else:
                passthrough[name] = t.contiguous()
            continue

        q, s = quantize_tensor_int8(t)
        quantized[name] = q
        scales[name] = s

    return {
        "quantized": quantized,
        "scales": scales,
        "passthrough": passthrough,
    }


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

def estimate_artifact_size(
    hnet: HyperNetwork,
    code_files: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Estimate the full artifact size:
      - Hypernetwork weights (int8 + zlib)
      - Code files
    """
    # 1. Quantize hypernetwork weights
    state_dict = hnet.state_dict()
    quant_obj = quantize_state_dict(state_dict)

    # Serialize + compress
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw_bytes = buf.getvalue()
    compressed = zlib.compress(raw_bytes, level=9)

    weights_raw = len(raw_bytes)
    weights_compressed = len(compressed)

    # 2. Code size
    code_total = 0
    if code_files is None:
        here = Path(__file__).parent
        hnet_dir = here / "hypernetwork"
        code_files_paths = [
            hnet_dir / "train_gpt_hypernet.py",  # single combined file is the artifact
        ]
    else:
        code_files_paths = [Path(f) for f in code_files]

    code_sizes = {}
    for f in code_files_paths:
        if f.exists():
            size = f.stat().st_size
            code_sizes[f.name] = size
            code_total += size

    # 3. Total
    total = weights_compressed + code_total

    if verbose:
        print("=" * 60)
        print("ARTIFACT SIZE REPORT")
        print("=" * 60)

        print(f"\n--- Hypernetwork Weights ---")
        print(f"  Parameters: {sum(p.numel() for p in hnet.parameters()):,}")
        print(f"  Raw state dict:  {weights_raw:>12,} bytes ({weights_raw / 1e6:.2f} MB)")
        print(f"  Int8 + zlib:     {weights_compressed:>12,} bytes ({weights_compressed / 1e6:.2f} MB)")
        print(f"  Compression:     {weights_raw / max(weights_compressed, 1):.2f}x")

        print(f"\n--- Code Files ---")
        for name, size in code_sizes.items():
            print(f"  {name}: {size:>8,} bytes")
        print(f"  Total code:      {code_total:>12,} bytes ({code_total / 1e6:.2f} MB)")

        print(f"\n--- Total Artifact ---")
        print(f"  Weights + Code:  {total:>12,} bytes ({total / 1e6:.2f} MB)")

        budget = 16_000_000
        if total <= budget:
            headroom = budget - total
            print(f"  STATUS: FITS! ({headroom:,} bytes headroom, {headroom/budget*100:.1f}%)")
        else:
            overage = total - budget
            print(f"  STATUS: OVER BUDGET by {overage:,} bytes ({overage/1e6:.2f} MB)")

        # Also show what the generated GPT would look like
        target_params = hnet.count_target_parameters()
        hnet_params = sum(p.numel() for p in hnet.parameters())
        print(f"\n--- Generation Stats ---")
        print(f"  HyperNet params:  {hnet_params:>12,}")
        print(f"  Target GPT params: {target_params:>12,}")
        print(f"  Expansion ratio:   {target_params / hnet_params:>11.2f}x")

    return {
        "weights_raw": weights_raw,
        "weights_compressed": weights_compressed,
        "code_total": code_total,
        "total": total,
        "budget": 16_000_000,
        "fits": total <= 16_000_000,
    }


# ---------------------------------------------------------------------------
# Sweep different hypernetwork configs
# ---------------------------------------------------------------------------

def sweep_configs():
    """Try different hypernetwork configurations and report sizes."""
    print("=" * 70)
    print("HYPERNETWORK CONFIG SWEEP")
    print("=" * 70)

    configs = [
        # (name, cond_dim, trunk_hidden, trunk_layers, chunk_size)
        ("tiny",     32,  256, 2, 2048),
        ("small",    48,  384, 2, 4096),
        ("medium",   64,  512, 3, 4096),
        ("large",    96,  768, 3, 4096),
        ("xlarge",  128, 1024, 3, 4096),
        ("wide",     64, 1024, 2, 4096),
        ("deep",     64,  512, 5, 4096),
        ("bigchunk", 64,  512, 3, 8192),
        ("smallchunk", 64, 512, 3, 2048),
    ]

    print(f"\n{'Name':<12} {'Cond':<5} {'Hidden':<7} {'Layers':<7} {'Chunk':<6} "
          f"{'HNet Params':<13} {'Target Params':<14} {'Ratio':<7} "
          f"{'Artifact MB':<12} {'Fits?':<6}")
    print("-" * 100)

    for name, cond_dim, hidden, layers, chunk in configs:
        cfg = HyperNetConfig(
            cond_dim=cond_dim,
            trunk_hidden=hidden,
            trunk_layers=layers,
            chunk_size=chunk,
        )
        hnet = HyperNetwork(cfg)
        result = estimate_artifact_size(hnet, verbose=False)
        hnet_params = sum(p.numel() for p in hnet.parameters())
        target_params = hnet.count_target_parameters()
        ratio = target_params / hnet_params

        status = "YES" if result["fits"] else "NO"
        print(f"{name:<12} {cond_dim:<5} {hidden:<7} {layers:<7} {chunk:<6} "
              f"{hnet_params:<13,} {target_params:<14,} {ratio:<7.2f} "
              f"{result['total']/1e6:<12.2f} {status:<6}")

    # Also sweep target GPT configs
    print(f"\n\n--- Different Target GPT sizes (medium hypernetwork) ---\n")
    print(f"{'Target':<20} {'Layers':<7} {'Dim':<5} {'MLP':<4} "
          f"{'Target Params':<14} {'HNet Params':<13} {'Artifact MB':<12} {'Fits?':<6}")
    print("-" * 90)

    target_configs = [
        ("baseline",       9,  512, 2),
        ("11L-3xMLP",     11,  512, 3),
        ("12L-3xMLP",     12,  512, 3),
        ("13L-3xMLP",     13,  512, 3),
        ("11L-4xMLP",     11,  512, 4),
        ("11L-dim576",    11,  576, 3),
        ("11L-dim640",    11,  640, 3),
        ("14L-3xMLP",     14,  512, 3),
    ]

    for name, nlayers, dim, mlp_mult in target_configs:
        tcfg = TargetGPTConfig(
            num_layers=nlayers,
            model_dim=dim,
            mlp_mult=mlp_mult,
        )
        cfg = HyperNetConfig(target_cfg=tcfg)
        hnet = HyperNetwork(cfg)
        result = estimate_artifact_size(hnet, verbose=False)
        hnet_params = sum(p.numel() for p in hnet.parameters())
        target_params = hnet.count_target_parameters()

        status = "YES" if result["fits"] else "NO"
        print(f"{name:<20} {nlayers:<7} {dim:<5} {mlp_mult:<4} "
              f"{target_params:<14,} {hnet_params:<13,} {result['total']/1e6:<12.2f} {status:<6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check artifact size for hypernetwork approach")
    parser.add_argument("--budget", type=int, default=16_000_000, help="Artifact budget in bytes")
    parser.add_argument("--sweep", action="store_true", help="Sweep different configurations")
    args = parser.parse_args()

    if args.sweep:
        sweep_configs()
    else:
        config = HyperNetConfig()
        hnet = HyperNetwork(config)
        result = estimate_artifact_size(hnet)
