"""
Pack selected chunks into a new .bin shard file for training.
Reads chunk_scores.json (or chunk_scores_trigram.json), extracts the
top chunks from the original shards, and writes them contiguously.
"""

import numpy as np
import json
import sys
from pathlib import Path
import time

DATA_DIR = Path("data/datasets/fineweb10B_sp1024")
OUTPUT_DIR = Path("experiments/data_order/stage2_chunk_level")
VOCAB_SIZE = 1024


def load_shard_tokens(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520 and int(header[1]) == 1
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)


def write_shard(path, tokens):
    """Write tokens in the same binary format as the original shards."""
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(tokens)
    with open(path, "wb") as f:
        header.tofile(f)
        tokens.astype("<u2").tofile(f)
    print(f"  Wrote {path}: {len(tokens):,} tokens ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    t_start = time.time()

    # Choose score file: bigram or trigram
    score_file = sys.argv[1] if len(sys.argv) > 1 else "chunk_scores.json"
    budget = sys.argv[2] if len(sys.argv) > 2 else "1gpu"

    with open(OUTPUT_DIR / score_file) as f:
        data = json.load(f)

    chunk_size = data["chunk_size"]
    selection_key = f"selection_{budget}"

    selection = data[selection_key]
    chunks_to_extract = selection["chunks"]
    print(f"Score file: {score_file}, budget: {budget}")
    print(f"Selecting {len(chunks_to_extract):,} chunks × {chunk_size} tokens")

    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))

    # Group chunks by shard for efficient loading
    by_shard = {}
    for entry in chunks_to_extract:
        shard_idx = entry[0]
        chunk_idx = entry[1]
        if shard_idx not in by_shard:
            by_shard[shard_idx] = []
        by_shard[shard_idx].append(chunk_idx)

    print(f"Chunks span {len(by_shard)} shards")

    # Extract chunks shard by shard
    all_tokens = []
    for shard_idx in sorted(by_shard.keys()):
        path = shard_paths[shard_idx]
        tokens = load_shard_tokens(path)
        chunk_indices = sorted(by_shard[shard_idx])

        for ci in chunk_indices:
            start = ci * chunk_size
            end = start + chunk_size
            all_tokens.append(tokens[start:end])

        if len(all_tokens) % 1000 < len(chunk_indices):
            print(f"  Extracted {len(all_tokens):,} chunks so far...")

    combined = np.concatenate(all_tokens)
    print(f"\nTotal tokens: {len(combined):,}")

    # Write as shards (100M tokens each, matching original format)
    # Include scorer name in dir to avoid overwriting different selections
    scorer_name = score_file.replace("chunk_scores_", "").replace("chunk_scores", "bigram").replace(".json", "")
    out_dir = DATA_DIR / f"fineweb10B_sp1024_chunk_{scorer_name}_{budget}"
    out_dir.mkdir(exist_ok=True)

    # Copy val shard
    val_src = DATA_DIR / "fineweb_val_000000.bin"
    val_dst = out_dir / "fineweb_val_000000.bin"
    if not val_dst.exists():
        import os
        os.symlink(val_src.resolve(), val_dst)

    # Write train shards
    tokens_per_shard = 100_000_000
    n_shards = (len(combined) + tokens_per_shard - 1) // tokens_per_shard

    for i in range(n_shards):
        start = i * tokens_per_shard
        end = min(start + tokens_per_shard, len(combined))
        shard_tokens = combined[start:end]
        shard_path = out_dir / f"fineweb_train_{i:06d}.bin"
        write_shard(shard_path, shard_tokens)

    print(f"\nWrote {n_shards} shard(s) to {out_dir}")
    print(f"Total time: {time.time()-t_start:.1f}s")
