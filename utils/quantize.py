"""Parameter budget calculator and quantization utilities."""

import math
import torch
import torch.nn as nn

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024  # 16 MB


def count_params(model: nn.Module) -> int:
    """Count total parameters (includes shared/tied params multiple times)."""
    return sum(p.numel() for p in model.parameters())


def count_unique_params(model: nn.Module) -> int:
    """Count unique parameters (shared/tied params counted once)."""
    seen = set()
    total = 0
    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += p.numel()
    return total


def estimate_artifact_size(model: nn.Module, bits_per_param: int = 16,
                           scale_bytes_per_row: int = 0,
                           header_bytes: int = 4096,
                           compression_ratio: float = 1.0) -> int:
    """Estimate artifact size in bytes.

    Args:
        model: The model to estimate size for.
        bits_per_param: Bits per parameter (16=fp16, 8=int8, 6=int6).
        scale_bytes_per_row: Extra bytes per row for quantization scales.
            For per-row int8: 2 bytes (fp16 scale). For int6: 2 bytes.
        header_bytes: Overhead for checkpoint format, config, etc.
        compression_ratio: Expected compression ratio from zlib (1.0 = no compression,
            0.9 = 10% smaller). Typical for quantized weights: 0.85-0.95.

    Returns:
        Estimated artifact size in bytes.
    """
    unique_params = count_unique_params(model)
    param_bytes = math.ceil(unique_params * bits_per_param / 8)

    # Count rows for scale overhead
    n_rows = 0
    if scale_bytes_per_row > 0:
        seen = set()
        for p in model.parameters():
            if p.data_ptr() not in seen and p.dim() >= 2:
                seen.add(p.data_ptr())
                n_rows += p.shape[0]
    scale_bytes = n_rows * scale_bytes_per_row

    raw_size = param_bytes + scale_bytes + header_bytes
    return int(raw_size * compression_ratio)


def check_budget(model: nn.Module, max_bytes: int = MAX_ARTIFACT_BYTES,
                 bits_per_param: int = 16) -> tuple[bool, dict]:
    """Check if model fits within artifact budget.

    Returns:
        (fits, details_dict)
    """
    unique = count_unique_params(model)
    total = count_params(model)

    configs = {
        "fp16": {"bits": 16, "scale": 0, "compression": 1.0},
        "int8": {"bits": 8, "scale": 2, "compression": 1.0},
        "int8_zlib": {"bits": 8, "scale": 2, "compression": 0.92},
        "int6": {"bits": 6, "scale": 2, "compression": 1.0},
        "int6_zlib": {"bits": 6, "scale": 2, "compression": 0.88},
    }

    estimates = {}
    for name, cfg in configs.items():
        size = estimate_artifact_size(
            model, cfg["bits"], cfg["scale"], compression_ratio=cfg["compression"]
        )
        estimates[name] = {
            "bytes": size,
            "mb": size / (1024 * 1024),
            "fits": size <= max_bytes,
        }

    # Use requested bits_per_param for the primary check
    primary_size = estimate_artifact_size(model, bits_per_param)

    details = {
        "total_params": total,
        "unique_params": unique,
        "shared_params": total - unique,
        "sharing_ratio": total / unique if unique > 0 else 0,
        "primary_estimate_bytes": primary_size,
        "primary_estimate_mb": primary_size / (1024 * 1024),
        "budget_mb": max_bytes / (1024 * 1024),
        "estimates": estimates,
    }

    return primary_size <= max_bytes, details


def print_budget_report(model: nn.Module, name: str = "Model"):
    """Print a formatted budget report."""
    fits, details = check_budget(model)
    print(f"\n{'='*60}")
    print(f"  {name} — Parameter Budget Report")
    print(f"{'='*60}")
    print(f"  Total params:  {details['total_params']:>12,}")
    print(f"  Unique params: {details['unique_params']:>12,}")
    if details['shared_params'] > 0:
        print(f"  Shared params: {details['shared_params']:>12,} "
              f"({details['sharing_ratio']:.1f}x reuse)")
    print(f"  Budget:        {details['budget_mb']:>12.2f} MB")
    print()
    print(f"  {'Format':<12} {'Size (MB)':>10} {'Fits?':>6}")
    print(f"  {'-'*30}")
    for name, est in details['estimates'].items():
        marker = "YES" if est['fits'] else "NO"
        print(f"  {name:<12} {est['mb']:>10.2f} {marker:>6}")
    print(f"{'='*60}\n")
