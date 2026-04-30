"""
build_ngram_oracle.py -- Offline n-gram oracle builder for Parameter Golf.

Scans all FineWeb training shards and builds order 1-8 n-gram hash tables,
stored as int8 log-probabilities, compressed with zstd-22.

Run BEFORE the 10-minute training clock:
    python build_ngram_oracle.py [shard_pattern] [output_path]

Defaults:
    shard_pattern = ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
    output_path   = ./ngram_oracle.bin
"""
from __future__ import annotations

import glob
import io
import math
import struct
import sys
import time

import numpy as np

try:
    import zstandard as zstd
    USE_ZSTD = True
except ImportError:
    import zlib
    USE_ZSTD = False

VOCAB = 1024
MAGIC = 0x4E474F52  # 'NGOR'
SCALE = 10.0 / 127.0  # int8 -> log-nats mapping

# Bucket counts per order. Tuned so total compressed oracle fits in ~1.5-2 MB.
# Order 1: exact unigram (1 row of 1024)
# Order 2: exact bigram (1024 rows of 1024)
# Orders 3-8: FNV1a hashed
BUCKET_CONFIGS: dict[int, int] = {
    1: 1,               # single row for unigram
    2: VOCAB,           # exact bigram: prev -> distribution
    3: 4096,
    4: 2048,
    5: 1024,
    6: 512,
    7: 256,
    8: 256,
}


def fnv1a_hash_np(context_columns: list[np.ndarray], buckets: int) -> np.ndarray:
    """FNV-1a hash over a list of token columns. Returns bucket indices."""
    h = np.full(len(context_columns[0]), np.uint32(2166136261), dtype=np.uint32)
    for col in context_columns:
        h ^= col.astype(np.uint32)
        h = (h * np.uint32(16777619)) & np.uint32(0xFFFFFFFF)
    return (h % np.uint32(buckets)).astype(np.int32)


def load_shard(path: str) -> np.ndarray:
    """Load a FineWeb .bin shard. Format: 256 int32 header, then uint16 tokens."""
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)
    return tokens.astype(np.int32)


def count_order(tokens: np.ndarray, order: int, buckets: int) -> np.ndarray:
    """Build count table for a given n-gram order from a token stream.
    Returns int32 array of shape (buckets, VOCAB)."""
    counts = np.zeros((buckets, VOCAB), dtype=np.int32)
    n = len(tokens)

    if order == 1:
        # Unigram: just count all tokens
        np.add.at(counts[0], tokens, 1)
        return counts

    if order == 2:
        # Exact bigram via chunked vectorized bincount.
        # Chunked to bound peak memory (full pairs[] at 100M tokens = ~800 MiB int64).
        flat = np.zeros(VOCAB * VOCAB, dtype=np.int64)
        CHUNK = 5_000_000
        n_pairs = len(tokens) - 1
        for start in range(0, n_pairs, CHUNK):
            end = min(start + CHUNK, n_pairs)
            prev = tokens[start:end].astype(np.int64)
            curr = tokens[start + 1:end + 1].astype(np.int64)
            pairs = prev * VOCAB + curr
            flat += np.bincount(pairs, minlength=VOCAB * VOCAB)
        return flat.reshape(VOCAB, VOCAB).astype(np.int32)

    # Higher orders: hash the (order-1) context tokens.
    # Chunked to bound peak memory (full ctx_cols at 100M tokens × ctx_len × uint32
    # would be ~400 MiB per column for order 3+).
    ctx_len = order - 1
    n_targets = len(tokens) - ctx_len
    flat_counts = np.zeros(buckets * VOCAB, dtype=np.int64)
    CHUNK = 2_000_000
    for start in range(0, n_targets, CHUNK):
        end = min(start + CHUNK, n_targets)
        chunk_ctx = [tokens[start + i:end + i] for i in range(ctx_len)]
        chunk_targets = tokens[start + ctx_len:end + ctx_len]
        bucket_ids = fnv1a_hash_np(chunk_ctx, buckets)
        flat = bucket_ids.astype(np.int64) * VOCAB + chunk_targets.astype(np.int64)
        flat_counts += np.bincount(flat, minlength=buckets * VOCAB)
    return flat_counts.reshape(buckets, VOCAB).astype(np.int32)


def counts_to_int8_logprobs(counts: np.ndarray) -> np.ndarray:
    """Convert int32 counts to int8 log-probabilities with Laplace smoothing.
    Maps log-probs in [-10, 0] to int8 range [-127, 0]."""
    c = counts.astype(np.float64) + 1.0  # Laplace smoothing
    if c.ndim == 1:
        lp = np.log(c / c.sum())
    else:
        lp = np.log(c / c.sum(axis=1, keepdims=True))
    lp_clamped = np.clip(lp, -10.0, 0.0)
    return np.round(lp_clamped / SCALE).astype(np.int8)


def build_oracle(shard_pattern: str, output_path: str) -> None:
    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"No shards matching: {shard_pattern}")

    print(f"Found {len(files)} training shards")
    print(f"Orders: {sorted(BUCKET_CONFIGS.keys())}")
    print(f"Buckets: {BUCKET_CONFIGS}")

    # Accumulate counts across all shards
    all_counts: dict[int, np.ndarray] = {
        order: np.zeros((buckets, VOCAB), dtype=np.int32)
        for order, buckets in BUCKET_CONFIGS.items()
    }

    t0 = time.time()
    total_tokens = 0
    for i, shard_path in enumerate(files):
        tokens = load_shard(shard_path)
        total_tokens += len(tokens)
        print(f"  [{i+1}/{len(files)}] {shard_path}: {len(tokens):,} tokens "
              f"(cumulative: {total_tokens:,})")

        for order, buckets in BUCKET_CONFIGS.items():
            if len(tokens) <= order:
                continue
            shard_counts = count_order(tokens, order, buckets)
            all_counts[order] += shard_counts

    elapsed = time.time() - t0
    print(f"\nCounting complete: {total_tokens:,} tokens in {elapsed:.1f}s")

    # Serialize: header + per-order int8 log-prob tables
    buf = io.BytesIO()
    buf.write(struct.pack("<II", MAGIC, len(BUCKET_CONFIGS)))

    raw_total = 8  # header
    for order in sorted(BUCKET_CONFIGS.keys()):
        data = counts_to_int8_logprobs(all_counts[order])
        rows, cols = data.shape
        buf.write(struct.pack("<III", order, rows, cols))
        buf.write(data.tobytes())
        table_bytes = rows * cols
        raw_total += 12 + table_bytes
        print(f"  Order {order}: {rows}x{cols} = {table_bytes:,} bytes")

    raw_bytes = buf.getvalue()

    # Compress
    if USE_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(raw_bytes)
        method = "zstd-22"
    else:
        compressed = zlib.compress(raw_bytes, level=9)
        method = "zlib-9"

    with open(output_path, "wb") as f:
        f.write(compressed)

    print(f"\nOracle built:")
    print(f"  Raw: {len(raw_bytes):,} bytes ({len(raw_bytes)/1024/1024:.3f} MB)")
    print(f"  Compressed ({method}): {len(compressed):,} bytes ({len(compressed)/1024/1024:.3f} MB)")
    print(f"  Ratio: {len(raw_bytes)/len(compressed):.1f}x")
    print(f"  Output: {output_path}")


def _self_test() -> None:
    """Sanity-check the FNV-1a hash against the same hash implemented in PyTorch.

    The oracle is built with NumPy uint32 FNV-1a; the lookup at training/eval time
    is reimplemented in torch.int64 + masking. If the two diverge for any context,
    every higher-order lookup silently returns wrong distributions. This test
    asserts they agree on a small random sample.

    Skipped silently if torch is not installed (the build step shouldn't require it).
    """
    try:
        import torch
    except ImportError:
        print("[self-test] torch not available; skipping cross-impl FNV-1a check")
        return

    rng = np.random.default_rng(42)
    n = 1000
    ctx_len = 5
    buckets = 4096
    tokens = rng.integers(0, VOCAB, size=(n, ctx_len), dtype=np.int32)

    # NumPy path (matches build_oracle's hashing)
    cols_np = [tokens[:, i] for i in range(ctx_len)]
    np_hash = fnv1a_hash_np(cols_np, buckets)

    # Torch path (matches FrozenNgramOracle._fnv1a_hash exactly)
    t = torch.from_numpy(tokens).to(torch.int64)
    h = torch.full((n,), 2166136261, dtype=torch.int64)
    for i in range(ctx_len):
        h = h ^ t[:, i]
        h = (h * 16777619) & 0xFFFFFFFF
    torch_hash = (h % buckets).numpy()

    mismatches = int((np_hash != torch_hash).sum())
    if mismatches > 0:
        diff_idx = np.where(np_hash != torch_hash)[0][:5]
        raise AssertionError(
            f"[self-test] FNV-1a NumPy/Torch mismatch on {mismatches}/{n} samples; "
            f"first diffs at {diff_idx}: np={np_hash[diff_idx]} torch={torch_hash[diff_idx]}"
        )
    print(f"[self-test] FNV-1a NumPy/Torch agree on {n} samples (ctx_len={ctx_len}, buckets={buckets})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
        sys.exit(0)
    _self_test()  # Always run the cross-impl check before building.
    shard_pattern = (
        sys.argv[1] if len(sys.argv) > 1
        else "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"
    )
    output = sys.argv[2] if len(sys.argv) > 2 else "./ngram_oracle.bin"
    build_oracle(shard_pattern, output)
