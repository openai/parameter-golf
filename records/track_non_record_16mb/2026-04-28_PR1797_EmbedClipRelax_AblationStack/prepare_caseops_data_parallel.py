"""Parallel version of prepare_caseops_data.py.

Splits the encode step across a multiprocessing pool. Output layout is
identical — val/train shards + byte sidecar. Keeps doc order (imap) so the
first ``--val-docs`` documents go to the validation set exactly as in the
serial version.

The encoding of one doc (encode_lossless_caps_v2 + sp.encode + original-byte
LUT) is the hot path. sp.SentencePieceProcessor is not fork-safe in a useful
way, so each worker loads its own copy of the model.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
from typing import Iterator

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lossless_caps import (  # noqa: E402
    LOSSLESS_CAPS_CASEOPS_V1,
    encode_lossless_caps_v2,
    surface_piece_original_byte_counts,
)


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_TOKENS = 10_000_000
BOS_ID = 1


def _write_shard(out_path: pathlib.Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


# Per-worker globals. Initialized in _worker_init (spawn context).
_SP = None


def _worker_init(sp_path: str) -> None:
    global _SP
    _SP = spm.SentencePieceProcessor(model_file=sp_path)


def _encode_doc(job: tuple[int, str]) -> tuple[int, list[int], list[int] | None]:
    """Encode a single doc. Return (doc_idx, token_ids, byte_counts_or_None).

    byte_counts is computed only for validation docs (caller signals via job
    index + a caller-side threshold).  Since we do not know val_docs here, we
    ALWAYS compute byte_counts — cheap, and caller drops train ones.
    """
    idx, text = job
    transformed = encode_lossless_caps_v2(text)
    token_ids = [BOS_ID] + _SP.encode(transformed, out_type=int)
    proto = _SP.encode_as_immutable_proto(transformed)
    byte_counts = list(
        surface_piece_original_byte_counts(
            (piece.surface for piece in proto.pieces),
            text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
        )
    )
    return idx, token_ids, byte_counts


def _iter_doc_jobs(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) else obj
            yield idx, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, type=pathlib.Path)
    ap.add_argument("--out",  required=True, type=pathlib.Path)
    ap.add_argument("--sp",   required=True, type=pathlib.Path)
    ap.add_argument("--val-docs", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--chunksize", type=int, default=128,
                    help="imap chunk size (docs per RPC to worker)")
    args = ap.parse_args()

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    print(f"workers={args.workers} chunksize={args.chunksize} val_docs={args.val_docs}", flush=True)

    val_buf_tokens: list[int] = []
    val_buf_bytes: list[int] = []
    train_buf: list[int] = []
    val_written = 0
    train_written = 0
    n_docs = 0
    t_log = __import__("time").time()

    with mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(args.sp),),
    ) as pool:
        for idx, token_ids, byte_counts in pool.imap(
            _encode_doc, _iter_doc_jobs(args.docs), chunksize=args.chunksize
        ):
            if n_docs < args.val_docs:
                val_buf_tokens.extend(token_ids)
                val_buf_bytes.append(0)  # BOS contributes 0 original bytes
                val_buf_bytes.extend(int(b) for b in byte_counts)
                if len(val_buf_tokens) >= SHARD_TOKENS:
                    _write_shard(train_out / f"fineweb_val_{val_written:06d}.bin",
                                 np.array(val_buf_tokens[:SHARD_TOKENS], dtype=np.uint16))
                    _write_shard(train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                                 np.array(val_buf_bytes[:SHARD_TOKENS], dtype=np.uint16))
                    val_buf_tokens = val_buf_tokens[SHARD_TOKENS:]
                    val_buf_bytes = val_buf_bytes[SHARD_TOKENS:]
                    val_written += 1
            else:
                train_buf.extend(token_ids)
                if len(train_buf) >= SHARD_TOKENS:
                    _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                                 np.array(train_buf[:SHARD_TOKENS], dtype=np.uint16))
                    train_buf = train_buf[SHARD_TOKENS:]
                    train_written += 1
            n_docs += 1
            if n_docs % 50_000 == 0:
                t2 = __import__("time").time()
                rate = 50_000 / max(t2 - t_log, 1e-6)
                print(f"  processed {n_docs} docs  train_shards={train_written}  val_shards={val_written}  rate={rate:.0f}/s", flush=True)
                t_log = t2

    if val_buf_tokens:
        _write_shard(train_out / f"fineweb_val_{val_written:06d}.bin",
                     np.array(val_buf_tokens, dtype=np.uint16))
        _write_shard(train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                     np.array(val_buf_bytes, dtype=np.uint16))
    if train_buf:
        _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                     np.array(train_buf, dtype=np.uint16))

    print(f"done. docs={n_docs} train_shards={train_written + (1 if train_buf else 0)} val_shards={val_written + (1 if val_buf_tokens else 0)}")


if __name__ == "__main__":
    main()
