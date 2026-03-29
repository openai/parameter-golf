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


def estimate_vram_bytes(
    model_dim: int,
    num_layers: int,
    mlp_mult: int,
    num_heads: int,
    num_kv_heads: int,
    vocab_size: int = 1024,
    batch_tokens: int = 65536,
    seq_len: int = 1024,
) -> float:
    """Estimate peak VRAM usage in bytes during training.

    Accounts for: model weights (fp32 + bf16 cached), optimizer state (2x fp32),
    activations (bf16), gradients (bf16), and n-gram hash tables.
    """
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

    # Weights: fp32 (4 bytes) + bf16 compile cache (2 bytes)
    weight_bytes = total_params * 6

    # Optimizer state: Muon momentum buffer (fp32) + Adam states (2x fp32 for embeds/scalars)
    # Rough: ~2.5x fp32 per param on average
    optimizer_bytes = total_params * 10

    # Gradients: bf16
    grad_bytes = total_params * 2

    # Activations: rough estimate per micro-batch
    # Each layer stores: input (bf16), attn output, mlp output, residuals
    batch_seqs = batch_tokens // seq_len // 8  # 8 grad accum steps
    batch_seqs = max(batch_seqs, 1)
    act_per_layer = batch_seqs * seq_len * model_dim * 2 * 4  # ~4 tensors bf16
    activation_bytes = num_layers * act_per_layer

    # N-gram hash tables (if enabled): 7 orders × hash_size × vocab × 4 bytes
    # hash_size=128K, vocab=1024 → 3.7GB. But only during eval.
    # During training this is 0.
    ngram_bytes = 0  # not during training

    # KV cache for eval: negligible vs training
    # Torch compile overhead, CUDA context: ~500MB fixed
    overhead = 500 * 1024 * 1024

    return (
        weight_bytes
        + optimizer_bytes
        + grad_bytes
        + activation_bytes
        + ngram_bytes
        + overhead
    )


def find_max_model(
    quant_bits: int,
    enable_entropy_coding: int,
    enable_pruning: int,
    prune_fraction: float,
    target_bytes: int = 16_000_000,
    vram_bytes: int = 24 * 1024 * 1024 * 1024,  # 24 GiB default (4090)
    batch_tokens: int = 65536,
) -> tuple[dict, int, float] | None:
    """Find the largest model config that fits in BOTH the artifact budget and VRAM.

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
                        vram = estimate_vram_bytes(
                            model_dim,
                            num_layers,
                            mlp_mult,
                            num_heads,
                            num_kv_heads,
                            batch_tokens=batch_tokens,
                        )
                        if (
                            size <= target_bytes
                            and vram <= vram_bytes
                            and (best is None or params > best[1])
                        ):
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
