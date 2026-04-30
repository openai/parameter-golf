"""One-off helper: build a tiny byte260 dataset for *local smoke testing*.

Streams FineWeb-Edu via HuggingFace, encodes UTF-8 bytes with the parameter-golf
PureByteTokenizer (vocab 260: 4 specials + 256 byte values, offset 4), writes
two shards in the parameter-golf .bin format (header magic 20240520, uint16
tokens). Output goes to data/datasets/fineweb10B_byte260/.

This is NOT the official byte260 export — that requires re-tokenizing the
exact docs_selected.jsonl from the parameter-golf manifest. Use this only for
proving train_hnet.py runs end-to-end. Real numbers go on Runpod.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
BYTE_OFFSET = 4

TRAIN_TARGET_BYTES = 4_000_000
VAL_TARGET_BYTES = 200_000


def encode_text(text: str) -> np.ndarray:
    raw = text.encode("utf-8", errors="replace")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16) + BYTE_OFFSET


def write_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype=np.int32)
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("missing huggingface datasets — pip install datasets", file=sys.stderr)
        raise

    out_dir = Path("data/datasets/fineweb10B_byte260")
    print(f"streaming FineWeb-Edu sample into {out_dir} ...", flush=True)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )

    val_chunks: list[np.ndarray] = []
    train_chunks: list[np.ndarray] = []
    val_bytes = train_bytes = 0

    for i, doc in enumerate(ds):
        toks = encode_text(doc["text"])
        if val_bytes < VAL_TARGET_BYTES:
            val_chunks.append(toks)
            val_bytes += toks.size
        elif train_bytes < TRAIN_TARGET_BYTES:
            train_chunks.append(toks)
            train_bytes += toks.size
        else:
            break
        if (i + 1) % 200 == 0:
            print(f"  doc {i+1}: val={val_bytes:,} train={train_bytes:,}", flush=True)

    train_arr = np.concatenate(train_chunks)
    val_arr = np.concatenate(val_chunks)
    print(f"final: train={train_arr.size:,} val={val_arr.size:,}")

    write_shard(out_dir / "fineweb_train_000000.bin", train_arr)
    write_shard(out_dir / "fineweb_val_000000.bin", val_arr)
    print(f"wrote shards to {out_dir}")


if __name__ == "__main__":
    main()
