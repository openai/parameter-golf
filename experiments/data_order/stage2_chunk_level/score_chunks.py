"""
Stage 2: Score every 32K-token chunk across all training shards.
Uses val-trained bigram LM (validated at rho=0.984 in Stage 1).
Outputs per-chunk CE scores for selection.
"""

import numpy as np
import json
import struct
from pathlib import Path
import time

DATA_DIR = Path("data/datasets/fineweb10B_sp1024")
VOCAB_SIZE = 1024
OUTPUT_DIR = Path("experiments/data_order/stage2_chunk_level")
CHUNK_SIZE = 32768  # 32K tokens per chunk


def load_shard_tokens(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520 and int(header[1]) == 1
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)


def bigram_counts_flat(tokens):
    prev = tokens[:-1].astype(np.int64)
    curr = tokens[1:].astype(np.int64)
    return np.bincount(prev * VOCAB_SIZE + curr, minlength=VOCAB_SIZE * VOCAB_SIZE).astype(np.float64)


def make_lm(bigram_2d, alpha=0.01):
    c = bigram_2d + alpha
    return np.log2(c / c.sum(axis=1, keepdims=True))


def chunk_ce_vectorized(val_lm_flat, tokens, chunk_size):
    """Score all chunks in a shard at once. Returns array of CE values."""
    n_chunks = len(tokens) // chunk_size
    if n_chunks == 0:
        return np.array([])

    # Trim to exact multiple of chunk_size
    trimmed = tokens[:n_chunks * chunk_size]

    # Compute bigram log-probs for entire shard at once
    prev = trimmed[:-1].astype(np.int64)
    curr = trimmed[1:].astype(np.int64)
    lp = val_lm_flat[prev * VOCAB_SIZE + curr]

    # Reshape into chunks and average (note: last bigram of each chunk
    # spans into next chunk, but at 32K this is negligible)
    # Each chunk has chunk_size-1 bigrams
    bigrams_per_chunk = chunk_size - 1
    # Trim the log-probs to exact multiple
    n_usable = n_chunks * bigrams_per_chunk
    lp_trimmed = lp[:n_usable].reshape(n_chunks, bigrams_per_chunk)
    ce_per_chunk = -lp_trimmed.mean(axis=1)

    return ce_per_chunk


if __name__ == "__main__":
    t_start = time.time()

    val_path = DATA_DIR / "fineweb_val_000000.bin"
    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
    print(f"Shards: {len(shard_paths)}, chunk_size: {CHUNK_SIZE}")

    # Train val bigram LM
    print("Training val bigram LM...")
    val_tokens = load_shard_tokens(val_path)
    val_bi = bigram_counts_flat(val_tokens)
    val_lm = make_lm(val_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))
    val_lm_flat = val_lm.flatten()
    print(f"Val: {len(val_tokens):,} tokens, LM ready")

    # Score all chunks
    all_scores = []  # (shard_idx, chunk_idx_within_shard, ce)
    all_ce = []  # flat list for histogram
    per_shard_stats = []

    for shard_idx, path in enumerate(shard_paths):
        tokens = load_shard_tokens(path)
        ces = chunk_ce_vectorized(val_lm_flat, tokens, CHUNK_SIZE)

        for chunk_idx, ce in enumerate(ces):
            all_scores.append((shard_idx, chunk_idx, float(ce)))
            all_ce.append(float(ce))

        per_shard_stats.append({
            "shard": path.stem,
            "n_chunks": len(ces),
            "mean_ce": float(ces.mean()),
            "std_ce": float(ces.std()),
            "min_ce": float(ces.min()),
            "max_ce": float(ces.max()),
        })

        if (shard_idx + 1) % 10 == 0:
            print(f"  {shard_idx+1}/{len(shard_paths)} shards, {len(all_ce)} chunks so far")

    all_ce = np.array(all_ce)
    print(f"\nScored {len(all_ce):,} chunks in {time.time()-t_start:.1f}s")

    # ── Distribution analysis ─────────────────────────────────────────
    print(f"\n=== CE Distribution ===")
    print(f"  Total chunks: {len(all_ce):,}")
    print(f"  Mean:   {all_ce.mean():.4f}")
    print(f"  Std:    {all_ce.std():.4f}")
    print(f"  Min:    {all_ce.min():.4f}")
    print(f"  Max:    {all_ce.max():.4f}")
    print(f"  Range:  {all_ce.max() - all_ce.min():.4f}")

    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}:    {np.percentile(all_ce, p):.4f}")

    # Within-shard vs between-shard variance
    shard_means = np.array([s["mean_ce"] for s in per_shard_stats])
    between_var = shard_means.var()
    within_vars = [s["std_ce"]**2 for s in per_shard_stats]
    within_var = np.mean(within_vars)
    print(f"\n  Between-shard variance: {between_var:.6f}")
    print(f"  Within-shard variance:  {within_var:.6f}")
    print(f"  Ratio (within/between): {within_var/between_var:.1f}x")

    # ── Selection ─────────────────────────────────────────────────────
    # Sort by CE ascending (lower = more val-like)
    all_scores.sort(key=lambda x: x[2])

    # How many chunks for different training budgets?
    tokens_1gpu = 1836 * 524288  # ~963M tokens (H100 single GPU, baseline step count)
    tokens_8gpu = 7185 * 524288  # ~3.77B tokens (8×H100)
    chunks_1gpu = tokens_1gpu // CHUNK_SIZE
    chunks_8gpu = tokens_8gpu // CHUNK_SIZE

    print(f"\n=== Selection ===")
    print(f"  1×GPU budget: {chunks_1gpu:,} chunks ({tokens_1gpu/1e9:.2f}B tokens)")
    print(f"  8×GPU budget: {chunks_8gpu:,} chunks ({tokens_8gpu/1e9:.2f}B tokens)")

    # CE stats for selected subsets
    for label, n_chunks in [("1×GPU", chunks_1gpu), ("8×GPU", chunks_8gpu)]:
        selected_ces = np.array([s[2] for s in all_scores[:n_chunks]])
        excluded_ces = np.array([s[2] for s in all_scores[n_chunks:]])
        print(f"\n  {label} selected ({n_chunks:,} chunks):")
        print(f"    CE range: {selected_ces.min():.4f} – {selected_ces.max():.4f}")
        print(f"    CE mean:  {selected_ces.mean():.4f} (vs full mean {all_ce.mean():.4f})")
        print(f"    CE gain:  {all_ce.mean() - selected_ces.mean():.4f} bits lower than average")

    # Show which shards contribute most to top selection
    print(f"\n  Shard contribution to top {chunks_1gpu} chunks (1×GPU):")
    shard_counts = {}
    for shard_idx, chunk_idx, ce in all_scores[:chunks_1gpu]:
        name = shard_paths[shard_idx].stem
        shard_counts[name] = shard_counts.get(name, 0) + 1
    for name, count in sorted(shard_counts.items(), key=lambda x: -x[1])[:15]:
        total = next(s["n_chunks"] for s in per_shard_stats if s["shard"] == name)
        print(f"    {name}: {count}/{total} chunks ({100*count/total:.0f}%)")

    # ── Save results ──────────────────────────────────────────────────
    # Save compact: just (shard_idx, chunk_idx, ce) sorted by CE
    with open(OUTPUT_DIR / "chunk_scores.json", "w") as f:
        json.dump({
            "chunk_size": CHUNK_SIZE,
            "total_chunks": len(all_scores),
            "distribution": {
                "mean": round(float(all_ce.mean()), 4),
                "std": round(float(all_ce.std()), 4),
                "min": round(float(all_ce.min()), 4),
                "max": round(float(all_ce.max()), 4),
                "between_shard_var": round(float(between_var), 6),
                "within_shard_var": round(float(within_var), 6),
            },
            "per_shard_stats": per_shard_stats,
            # Top 1×GPU selection
            "selection_1gpu": {
                "n_chunks": chunks_1gpu,
                "ce_range": [round(all_scores[0][2], 4), round(all_scores[chunks_1gpu-1][2], 4)],
                "chunks": [(s, c, round(ce, 4)) for s, c, ce in all_scores[:chunks_1gpu]],
            },
            # Top 8×GPU selection
            "selection_8gpu": {
                "n_chunks": chunks_8gpu,
                "ce_range": [round(all_scores[0][2], 4), round(all_scores[chunks_8gpu-1][2], 4)],
                "chunks": [(s, c, round(ce, 4)) for s, c, ce in all_scores[:chunks_8gpu]],
            },
        }, f)

    print(f"\nSaved to {OUTPUT_DIR / 'chunk_scores.json'}")
    print(f"Total time: {time.time()-t_start:.1f}s")
