"""Model size estimation and budget-filling."""

from __future__ import annotations


def estimate_compressed_size(
    model_dim: int,
    num_layers: int,
    mlp_mult: int,
    num_heads: int,
    num_kv_heads: int,
    quant_bits: int,
    enable_entropy_coding: int,
    enable_pruning: int,
    prune_fraction: float,
    vocab_size: int = 1024,
) -> tuple[float, int]:
    """Return (estimated_bytes, total_params)."""
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_hidden = mlp_mult * model_dim
    embed_params = vocab_size * model_dim
    per_layer = (
        model_dim * model_dim
        + model_dim * kv_dim * 2
        + model_dim * model_dim
        + model_dim * mlp_hidden
        + mlp_hidden * model_dim
        + model_dim * 5
        + num_heads
    )
    total_params = embed_params + num_layers * per_layer + model_dim * num_layers
    weight_2d = num_layers * per_layer * 0.95
    weight_1d = num_layers * per_layer * 0.05 + embed_params
    raw_bytes = (weight_2d * quant_bits / 8) + (weight_1d * 8 / 8)
    if enable_pruning:
        raw_bytes *= 1 - prune_fraction * 0.5
    compression_ratio = 0.70 if enable_entropy_coding else 0.75
    return raw_bytes * compression_ratio + 82_000, total_params


def find_max_model(
    quant_bits: int,
    enable_entropy_coding: int,
    enable_pruning: int,
    prune_fraction: float,
    target_bytes: int = 16_000_000,
) -> tuple[dict, int, float] | None:
    """Find the largest model config that fits in the budget.

    Returns (arch_dict, total_params, estimated_size) or None.
    """
    best = None
    for num_layers in [8, 9, 10, 11, 12, 13, 14]:
        for model_dim in [384, 448, 512, 576, 640, 768]:
            for mlp_mult in [2, 3]:
                for num_heads in [4, 8]:
                    if model_dim % num_heads != 0:
                        continue
                    for num_kv_heads in [2, 4]:
                        if num_heads % num_kv_heads != 0:
                            continue
                        size, params = estimate_compressed_size(
                            model_dim,
                            num_layers,
                            mlp_mult,
                            num_heads,
                            num_kv_heads,
                            quant_bits,
                            enable_entropy_coding,
                            enable_pruning,
                            prune_fraction,
                        )
                        if size <= target_bytes and (best is None or params > best[1]):
                            best = (
                                dict(
                                    num_layers=num_layers,
                                    model_dim=model_dim,
                                    mlp_mult=mlp_mult,
                                    num_heads=num_heads,
                                    num_kv_heads=num_kv_heads,
                                ),
                                params,
                                size,
                            )
    return best
