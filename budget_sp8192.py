from __future__ import annotations

import argparse
import math


def estimate(vocab_size: int, model_dim: int, num_layers: int, num_heads: int, num_kv_heads: int,
             mlp_mult: float, bigram_vocab: int, bigram_dim: int, ve_enabled: bool, ve_dim: int,
             engram_enabled: bool, engram_vocab: int, engram_heads: int, engram_dim: int,
             engram_max_n: int, code_bytes: int, compression_ratio: float) -> dict[str, float]:
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_dim = int(mlp_mult * model_dim)
    tok = vocab_size * model_dim
    banks = (
        2 * num_layers * model_dim * model_dim
        + 2 * num_layers * kv_dim * model_dim
        + num_layers * mlp_dim * model_dim
        + num_layers * model_dim * mlp_dim
    )
    small = num_layers * (6 * model_dim + num_heads) + max(num_layers // 2, num_layers - num_layers // 2) * model_dim
    bigram = 0
    if bigram_vocab > 0 and not engram_enabled:
        bigram = bigram_vocab * bigram_dim + (bigram_dim * model_dim if bigram_dim != model_dim else 0) + 1
    ve = 0
    if ve_enabled:
        ve = vocab_size * ve_dim + (ve_dim * kv_dim if ve_dim != kv_dim else 0) + 3
    engram = 0
    if engram_enabled:
        engram = (engram_max_n - 1) * engram_vocab * engram_dim
        engram += model_dim * engram_dim * (engram_max_n - 1)
        engram += engram_dim * (engram_max_n - 1) * model_dim
        engram += 1
    fp16_bytes = 2 * (tok + bigram + ve + engram + small)
    int6_bytes = 0.75 * banks
    quant_bytes_est = (fp16_bytes + int6_bytes) / compression_ratio
    total_est = quant_bytes_est + code_bytes
    return {
        "tok_params": tok,
        "bank_params": banks,
        "bigram_params": bigram,
        "ve_params": ve,
        "engram_params": engram,
        "small_params": small,
        "total_params": tok + banks + bigram + ve + engram + small,
        "estimated_model_bytes": quant_bytes_est,
        "estimated_total_bytes": total_est,
        "estimated_total_mb": total_est / 1_000_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--code-bytes", type=int, default=120_000)
    parser.add_argument("--compression-ratio", type=float, default=1.0)
    parser.add_argument("--engram-enabled", action="store_true")
    parser.add_argument("--engram-vocab", type=int, default=4096)
    parser.add_argument("--engram-heads", type=int, default=4)
    parser.add_argument("--engram-dim", type=int, default=64)
    parser.add_argument("--engram-max-n", type=int, default=3)
    args = parser.parse_args()
    candidates = [
        {"model_dim": 512, "num_layers": 11, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3.0, "ve_enabled": False},
        {"model_dim": 384, "num_layers": 10, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3.0, "ve_enabled": False},
        {"model_dim": 384, "num_layers": 9, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 2.5, "ve_enabled": False},
        {"model_dim": 320, "num_layers": 11, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3.0, "ve_enabled": False},
        {"model_dim": 320, "num_layers": 10, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3.0, "ve_enabled": False},
        {"model_dim": 384, "num_layers": 10, "num_heads": 8, "num_kv_heads": 2, "mlp_mult": 3.0, "ve_enabled": False},
    ]
    for c in candidates:
        r = estimate(
            vocab_size=args.vocab_size,
            bigram_vocab=0,
            bigram_dim=112,
            ve_dim=128,
            engram_enabled=args.engram_enabled,
            engram_vocab=args.engram_vocab,
            engram_heads=args.engram_heads,
            engram_dim=args.engram_dim,
            engram_max_n=args.engram_max_n,
            code_bytes=args.code_bytes,
            compression_ratio=args.compression_ratio,
            **c,
        )
        label = f"d{c['model_dim']}_L{c['num_layers']}_kv{c['num_kv_heads']}_mlp{c['mlp_mult']}"
        status = "FIT" if r["estimated_total_bytes"] <= 16_000_000 else "OVER"
        print(
            f"{label:24s} {status:4s} total={r['estimated_total_mb']:.2f}MB "
            f"params={r['total_params']/1_000_000:.2f}M tok={r['tok_params']/1_000_000:.2f}M "
            f"banks={r['bank_params']/1_000_000:.2f}M engram={r['engram_params']/1_000_000:.2f}M"
        )


if __name__ == "__main__":
    main()
