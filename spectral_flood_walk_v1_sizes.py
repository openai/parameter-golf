#!/usr/bin/env python3
from __future__ import annotations


def parse_int_csv(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def estimate_v1a_sizes(
    *,
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    ff_mult: int,
    pos_buckets: int,
    semantic_layers: str,
    use_semantic_memory: bool,
    pk_num_subkeys: int,
    pk_key_dim: int,
    pk_code_dim: int,
) -> dict[str, float]:
    semantic_layer_ids = parse_int_csv(semantic_layers) if use_semantic_memory else ()
    standard_layers = num_layers - len(semantic_layer_ids)
    semantic_layers_count = len(semantic_layer_ids)
    ff_hidden = embed_dim * ff_mult

    embed_bytes = vocab_size * embed_dim
    pos_bytes = pos_buckets * embed_dim
    attn_bytes_per_layer = 4 * embed_dim * embed_dim
    ffn_bytes_per_standard = 2 * embed_dim * ff_hidden
    pk_bytes_per_layer = (
        2 * pk_num_subkeys * pk_key_dim
        + (pk_num_subkeys * pk_num_subkeys * pk_code_dim)
        + (2 * embed_dim * pk_key_dim)
        + (pk_code_dim * embed_dim)
        + (embed_dim * embed_dim)
    )
    compact_model_bytes = (
        embed_bytes
        + pos_bytes
        + standard_layers * (attn_bytes_per_layer + ffn_bytes_per_standard)
        + semantic_layers_count * (attn_bytes_per_layer + pk_bytes_per_layer)
    )
    expanded_semantic_bytes = semantic_layers_count * (pk_num_subkeys * pk_num_subkeys * embed_dim * 2)
    return {
        "compact_model_bytes_estimate": float(compact_model_bytes),
        "compact_model_mb_estimate": float(compact_model_bytes / 1_000_000.0),
        "expanded_semantic_bytes_estimate": float(expanded_semantic_bytes),
        "expanded_semantic_mb_estimate": float(expanded_semantic_bytes / 1_000_000.0),
        "standard_layer_count": float(standard_layers),
        "semantic_layer_count": float(semantic_layers_count),
        "pk_entries_per_layer": float(pk_num_subkeys * pk_num_subkeys),
        "pk_code_bytes_per_layer": float(pk_num_subkeys * pk_num_subkeys * pk_code_dim),
    }
