"""Parallel rebuild of prepare_caseops_data.py — multiprocessing tokenization.

Bottleneck of the original is single-threaded SP encode + per-char byte
prefix-sum on val docs. With N workers via mp.Pool we get ~N× speedup.
On 28-vCPU pod, 16 workers cuts ~12h to ~45 min.

Same CLI as prepare_caseops_data.py + extra --workers flag.
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
from typing import Optional

import numpy as np
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lossless_caps import (
    encode_lossless_caps_v2,
    DEFAULT_V2_TITLE,
    DEFAULT_V2_ALLCAPS,
    DEFAULT_V2_CAPNEXT,
    DEFAULT_V2_ESC,
)

BOS_ID = 1
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_TOKENS = 10_000_000

_LOSSLESS_V2_OPERATORS_CHARS: Optional[frozenset] = None
_worker_sp: Optional[spm.SentencePieceProcessor] = None


def _make_operators_set() -> frozenset:
    return frozenset((
        DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC,
    ))


def _worker_init(sp_path: str) -> None:
    global _worker_sp, _LOSSLESS_V2_OPERATORS_CHARS
    _worker_sp = spm.SentencePieceProcessor(model_file=sp_path)
    _LOSSLESS_V2_OPERATORS_CHARS = _make_operators_set()


def _byte_counts(transformed: str, piece_ids: list, pieces: list) -> np.ndarray:
    n_chars = len(transformed)
    prefix = np.zeros(n_chars + 1, dtype=np.int64)
    running = 0
    ops = _LOSSLESS_V2_OPERATORS_CHARS
    for idx, ch in enumerate(transformed):
        if ch not in ops:
            cp = ord(ch)
            if cp < 0x80:
                running += 1
            elif cp < 0x800:
                running += 2
            elif cp < 0x10000:
                running += 3
            else:
                running += 4
        prefix[idx + 1] = running
    counts = np.empty(len(piece_ids), dtype=np.uint16)
    cursor_t = 0
    for i, piece in enumerate(pieces):
        surface = piece.replace("▁", " ")
        span_len = len(surface)
        end = cursor_t + span_len
        if end > n_chars:
            end = n_chars
        original_bytes = int(prefix[end] - prefix[cursor_t])
        cursor_t = end
        counts[i] = max(0, min(65535, original_bytes))
    return counts


def _worker_process_doc(args: tuple) -> tuple:
    """Worker: transform + tokenize one doc. Returns (doc_idx, token_ids, byte_counts_or_None)."""
    doc_idx, text, is_val = args
    sp = _worker_sp
    transformed = encode_lossless_caps_v2(text)
    piece_ids = sp.encode(transformed, out_type=int)
    token_ids = [BOS_ID] + piece_ids
    byte_counts = None
    if is_val:
        pieces = [sp.id_to_piece(int(pid)) for pid in piece_ids]
        byte_counts = _byte_counts(transformed, piece_ids, pieces)
    return doc_idx, token_ids, byte_counts


def _write_shard(path: pathlib.Path, arr: np.ndarray) -> None:
    # 256 int32 header = 1024 bytes — matches kevclark/parameter-golf format
    # and the load_data_shard expectations in train_gpt.py.
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = arr.size
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _iter_docs(docs_path: pathlib.Path):
    with docs_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield idx, (obj["text"] if isinstance(obj, dict) else obj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--sp", required=True, type=pathlib.Path)
    # Default 50,000 matches the canonical romeerp/parameter-golf-caseops-v1
    # split (docs_val=50000). The original default of 10,000 left docs
    # 10K-49,999 in train AND in canonical val — an 80% leak. Flagged in the
    # CaseOps memory-leakage audit (caseops-memory-leakage/, 2026-05).
    ap.add_argument("--val-docs", type=int, default=50_000)
    ap.add_argument("--max-docs", type=int, default=0, help="Stop after N total docs (0 = process all). Limits disk usage.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    print(f"loading sp model: {args.sp}", flush=True)
    sp_master = spm.SentencePieceProcessor(model_file=str(args.sp))
    print(f"loaded sp: vocab={sp_master.vocab_size()}", flush=True)
    print(f"workers: {args.workers}", flush=True)

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    val_buf_tokens: list = []
    val_buf_bytes: list = []
    train_buf: list = []
    val_written = 0
    train_written = 0
    n_docs = 0

    def _build_args():
        for doc_idx, text in _iter_docs(args.docs):
            if args.max_docs > 0 and doc_idx >= args.max_docs:
                return
            yield (doc_idx, text, doc_idx < args.val_docs)

    with mp.Pool(args.workers, initializer=_worker_init, initargs=(str(args.sp),)) as pool:
        for doc_idx, token_ids, byte_counts in pool.imap(
            _worker_process_doc,
            _build_args(),
            chunksize=args.chunksize,
        ):
            if doc_idx < args.val_docs:
                val_buf_tokens.extend(token_ids)
                val_buf_bytes.append(0)
                val_buf_bytes.extend(int(b) for b in byte_counts[: len(token_ids) - 1])
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
            if n_docs % 10_000 == 0:
                print(f"  processed {n_docs} docs  train_shards={train_written}  val_shards={val_written}", flush=True)

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
    mp.set_start_method("spawn", force=True)
    main()
