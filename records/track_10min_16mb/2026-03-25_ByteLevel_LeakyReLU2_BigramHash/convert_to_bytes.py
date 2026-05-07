#!/usr/bin/env python3
"""Convert sp1024 tokenized FineWeb shards to raw UTF-8 byte shards.

Usage:
    python convert_to_bytes.py \
        --src data/datasets/fineweb10B_sp1024 \
        --dst data/datasets/fineweb10B_bytes \
        --tokenizer data/tokenizers/fineweb_1024_bpe.model

The output shards use the same binary format (header + uint16 values) as the
sp1024 originals, so the training script can load them with zero code changes.
Each uint16 value is a raw byte (0-255) instead of a BPE token id (0-1023).
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import sentencepiece as spm

HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype("<i4").itemsize
MAGIC = 20240520
VERSION = 1


def _init_worker(tokenizer_path: str) -> None:
    """Each worker loads its own SentencePiece instance (not picklable)."""
    global _sp  # noqa: PLW0603
    _sp = spm.SentencePieceProcessor()
    _sp.Load(tokenizer_path)


def _convert_shard(args: tuple[str, str]) -> tuple[str, int, int]:
    src, dst = args
    header = np.fromfile(src, dtype="<i4", count=HEADER_INTS)
    num_tokens = int(header[2])
    tokens = np.fromfile(src, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)

    text = _sp.Decode(tokens.tolist())
    byte_vals = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)

    out_header = np.zeros(HEADER_INTS, dtype="<i4")
    out_header[0] = MAGIC
    out_header[1] = VERSION
    out_header[2] = len(byte_vals)

    with open(dst, "wb") as f:
        f.write(out_header.tobytes())
        f.write(byte_vals.tobytes())

    return os.path.basename(src), num_tokens, len(byte_vals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/datasets/fineweb10B_sp1024", help="Source sp1024 shard directory")
    parser.add_argument("--dst", default="data/datasets/fineweb10B_bytes", help="Output byte shard directory")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model", help="SentencePiece model")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.src, "fineweb_*.bin")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {args.src}")

    tasks = [(s, os.path.join(args.dst, os.path.basename(s))) for s in shards]

    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for i, (name, ntok, nbytes) in enumerate(pool.imap_unordered(_convert_shard, tasks)):
            if i % 20 == 0 or i == len(tasks) - 1:
                print(f"[{i + 1}/{len(tasks)}] {name}: {ntok:,} tokens -> {nbytes:,} bytes ({nbytes / ntok:.2f}x)")

    print(f"\nDone in {time.time() - t0:.0f}s. Output: {args.dst} ({len(tasks)} shards)")


if __name__ == "__main__":
    main()