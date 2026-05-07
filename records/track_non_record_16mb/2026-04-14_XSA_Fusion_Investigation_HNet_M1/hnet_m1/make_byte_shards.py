"""Convert SP8192 token shards into UTF-8 byte shards.

Reads the baseline's cached SP8192 bin files, decodes tokens to text via the
SentencePiece model, re-encodes to UTF-8 bytes, and writes byte shards in the
same on-disk layout (256 int32 header ints + <u2 payload, where u2 here stores
u16 values in [0, 255]).

We keep u16 payload (rather than u8) so the existing `load_data_shard` code
path in bigbag's baseline could be reused with minimal modification. Byte
values are in [0, 255], stored as u16.

Usage:
    python make_byte_shards.py \
        --tokenizer /path/to/fineweb_8192_bpe.model \
        --in-pattern '/path/to/fineweb10B_sp8192/fineweb_*_*.bin' \
        --out-dir /path/to/fineweb10B_bytes/
"""
from __future__ import annotations
import argparse, glob, sys, time
from pathlib import Path

import numpy as np
import sentencepiece as spm


SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def read_sp_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == SHARD_MAGIC and int(header[1]) == SHARD_VERSION, \
        f"bad header {path}"
    n = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=n, offset=256 * 4)
    assert tokens.size == n, f"short read {path}"
    return tokens


def write_byte_shard(path: Path, bytes_arr: np.ndarray) -> None:
    assert bytes_arr.dtype == np.uint16
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = bytes_arr.size
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(bytes_arr.tobytes())


def decode_tokens_to_bytes(sp: spm.SentencePieceProcessor, tokens: np.ndarray, chunk_size: int = 1_000_000) -> np.ndarray:
    """Decode a long token stream to UTF-8 bytes, processing in chunks.

    Returns a uint16 array of byte values in [0, 255].
    """
    out_pieces: list[bytes] = []
    n = tokens.size
    for i in range(0, n, chunk_size):
        chunk = tokens[i : i + chunk_size]
        text = sp.decode(chunk.tolist())
        b = text.encode("utf-8", errors="replace")
        out_pieces.append(b)
        if i % (chunk_size * 10) == 0:
            so_far = sum(len(p) for p in out_pieces)
            print(f"    decoded {i:>10d}/{n} tokens -> {so_far:>12d} bytes", flush=True)
    joined = b"".join(out_pieces)
    return np.frombuffer(joined, dtype=np.uint8).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Path to SentencePiece .model")
    ap.add_argument("--in-pattern", required=True, help="glob for input SP token shards")
    ap.add_argument("--out-dir", required=True, help="output dir for byte shards")
    ap.add_argument("--limit-shards", type=int, default=0, help="process only the first N shards (0 = all)")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"tokenizer vocab_size={sp.vocab_size()}", flush=True)

    in_paths = sorted(Path(p) for p in glob.glob(args.in_pattern))
    if args.limit_shards > 0:
        in_paths = in_paths[: args.limit_shards]
    print(f"input shards: {len(in_paths)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, inp in enumerate(in_paths):
        t0 = time.perf_counter()
        out_name = inp.name  # keep filename for dataloader compat
        out_path = out_dir / out_name
        if out_path.exists():
            # Assume valid and skip
            print(f"[{idx+1}/{len(in_paths)}] skip existing {out_path}", flush=True)
            continue
        print(f"[{idx+1}/{len(in_paths)}] reading {inp}", flush=True)
        toks = read_sp_shard(inp)
        print(f"    tokens: {toks.size}", flush=True)
        bytes_arr = decode_tokens_to_bytes(sp, toks)
        print(f"    -> {bytes_arr.size} bytes  ({bytes_arr.size / max(toks.size, 1):.2f} bytes/token)", flush=True)
        write_byte_shard(out_path, bytes_arr)
        print(f"    wrote {out_path}  in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
