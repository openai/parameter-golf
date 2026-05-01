"""Prepare CaseOps-tokenized FineWeb shards + per-token byte sidecar (parallel).

CaseOps (``lossless_caps_caseops_v1``) is a bijective, character-level text
transform that introduces four operator tokens in place of explicit
capitalization: TITLE, ALLCAPS, CAPNEXT, ESC. The transform is fully
reversible — no information is lost relative to the untransformed UTF-8
text, so BPB stays computable on TRUE byte counts.

Forward pipeline:
  1. Read the canonical FineWeb-10B doc stream (``docs_selected.jsonl``;
     fetched via ``download_docs.py``).
  2. Apply ``encode_lossless_caps_v2`` (the caseops_v1 alias) to each doc.
  3. Tokenize with the shipped SP model
     ``tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model``
     (reserves TITLE/ALLCAPS/CAPNEXT/ESC + sentinel as user_defined_symbols).
  4. Write uint16 train/val shards (``fineweb_{train,val}_XXXXXX.bin``).
  5. For the VAL stream only, emit per-token byte sidecar shards
     (``fineweb_val_bytes_XXXXXX.bin``, uint16 parallel arrays) that record
     each token's ORIGINAL pre-transform UTF-8 byte count. BPB is computed
     from these canonical bytes so the score is on the untransformed text
     (not the transformed representation).

Parallelism: per-doc (transform + SP tokenize + optional byte sidecar) is
handed to a multiprocessing.Pool. Doc order is preserved so the val/train
split (first ``--val-docs`` documents go to val) is identical to the
sequential reference. Workers each load their own SP processor.

Output layout — matches what ``train_gpt.py`` expects under
``DATA_DIR=./data`` with ``CASEOPS_ENABLED=1``:

    <out>/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
      fineweb_train_000000.bin
      fineweb_train_000001.bin
      ...
      fineweb_val_000000.bin
      fineweb_val_bytes_000000.bin

Usage:

    python3 prepare_caseops_data.py \\
        --docs ./data/datasets/docs_selected.jsonl \\
        --out  ./data \\
        --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
        [--workers N]    # default: os.cpu_count()
        [--chunksize N]  # default: 64

Requirements: sentencepiece, numpy. CPU-only. Runs once; reused across seeds.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import sentencepiece as spm

# Local import — lossless_caps.py ships next to this script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lossless_caps import (  # noqa: E402
    LOSSLESS_CAPS_CASEOPS_V1,
    encode_lossless_caps_v2,
    surface_piece_original_byte_counts,
)


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_TOKENS = 10_000_000  # tokens per shard — matches the main pipeline
BOS_ID = 1  # SP model's <s> control token; train_gpt.py:_find_docs requires BOS per doc


# Per-worker globals (initialized once per child process).
_worker_sp: Optional[spm.SentencePieceProcessor] = None
_worker_val_threshold: int = 0


def _worker_init(sp_path: str, val_threshold: int) -> None:
    global _worker_sp, _worker_val_threshold
    _worker_sp = spm.SentencePieceProcessor(model_file=sp_path)
    _worker_val_threshold = val_threshold


def _process_doc(args):
    """Tokenize one doc. Returns (idx, token_ids, byte_counts_or_None)."""
    idx, text = args
    transformed = encode_lossless_caps_v2(text)
    token_ids = [BOS_ID] + _worker_sp.encode(transformed, out_type=int)
    if idx < _worker_val_threshold:
        proto = _worker_sp.encode_as_immutable_proto(transformed)
        byte_counts = list(surface_piece_original_byte_counts(
            (piece.surface for piece in proto.pieces),
            text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
        ))
        return idx, token_ids, byte_counts
    return idx, token_ids, None


def _write_shard(out_path: pathlib.Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _iter_docs(docs_path: pathlib.Path):
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj


def _enumerated_docs(docs_path: pathlib.Path):
    for idx, text in enumerate(_iter_docs(docs_path)):
        yield idx, text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=pathlib.Path, help="Path to docs_selected.jsonl")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="Output base dir; shards go to <out>/datasets/...")
    ap.add_argument("--sp", required=True, type=pathlib.Path, help="Path to CaseOps SP model")
    ap.add_argument("--val-docs", type=int, default=10_000, help="Validation docs count (default: 10000)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Worker processes (default: os.cpu_count())")
    ap.add_argument("--chunksize", type=int, default=64, help="imap chunksize (default: 64)")
    args = ap.parse_args()

    # Verify SP model loads in main process (catch errors early).
    sp_main = spm.SentencePieceProcessor(model_file=str(args.sp))
    vocab_size = sp_main.vocab_size()
    del sp_main
    print(
        f"[main] sp_vocab={vocab_size} workers={args.workers} chunksize={args.chunksize} "
        f"val_docs={args.val_docs}",
        flush=True,
    )

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    val_buf_tokens: list[int] = []
    val_buf_bytes: list[int] = []
    train_buf: list[int] = []
    val_written = 0
    train_written = 0
    n_docs = 0
    t0 = time.time()
    last_t = t0
    last_n = 0

    with mp.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(str(args.sp), args.val_docs),
    ) as pool:
        for idx, token_ids, byte_counts in pool.imap(
            _process_doc,
            _enumerated_docs(args.docs),
            chunksize=args.chunksize,
        ):
            if byte_counts is not None:
                val_buf_tokens.extend(token_ids)
                val_buf_bytes.append(0)  # BOS contributes 0 original bytes
                val_buf_bytes.extend(int(b) for b in byte_counts)
                while len(val_buf_tokens) >= SHARD_TOKENS:
                    _write_shard(train_out / f"fineweb_val_{val_written:06d}.bin",
                                 np.array(val_buf_tokens[:SHARD_TOKENS], dtype=np.uint16))
                    _write_shard(train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                                 np.array(val_buf_bytes[:SHARD_TOKENS], dtype=np.uint16))
                    val_buf_tokens = val_buf_tokens[SHARD_TOKENS:]
                    val_buf_bytes = val_buf_bytes[SHARD_TOKENS:]
                    val_written += 1
            else:
                train_buf.extend(token_ids)
                while len(train_buf) >= SHARD_TOKENS:
                    _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                                 np.array(train_buf[:SHARD_TOKENS], dtype=np.uint16))
                    train_buf = train_buf[SHARD_TOKENS:]
                    train_written += 1
            n_docs += 1
            if n_docs % 50_000 == 0:
                now = time.time()
                rate = (n_docs - last_n) / max(now - last_t, 0.01)
                avg_rate = n_docs / max(now - t0, 0.01)
                print(
                    f"  [{n_docs:>10,}] train={train_written} val={val_written}  "
                    f"rate={rate:>7.0f} docs/s  avg={avg_rate:>7.0f} docs/s",
                    flush=True,
                )
                last_t = now
                last_n = n_docs

    # Flush tail buffers.
    if val_buf_tokens:
        _write_shard(train_out / f"fineweb_val_{val_written:06d}.bin",
                     np.array(val_buf_tokens, dtype=np.uint16))
        _write_shard(train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                     np.array(val_buf_bytes, dtype=np.uint16))
    if train_buf:
        _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                     np.array(train_buf, dtype=np.uint16))

    elapsed = time.time() - t0
    avg = n_docs / max(elapsed, 0.01)
    print(
        f"\ndone. docs={n_docs:,} elapsed={elapsed:.1f}s ({elapsed/60:.1f}m) "
        f"avg_rate={avg:.0f} docs/s "
        f"train_shards={train_written + (1 if train_buf else 0)} "
        f"val_shards={val_written + (1 if val_buf_tokens else 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
