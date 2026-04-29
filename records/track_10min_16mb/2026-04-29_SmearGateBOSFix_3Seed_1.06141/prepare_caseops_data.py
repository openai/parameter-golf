"""Prepare CaseOps-tokenized FineWeb shards + per-token byte sidecar.

CaseOps (``lossless_caps_caseops_v1``) is a bijective, character-level text
transform that introduces four operator tokens in place of explicit
capitalization: TITLE, ALLCAPS, CAPNEXT, ESC. The transform is fully
reversible — no information is lost relative to the untransformed UTF-8
text, so BPB stays computable on TRUE byte counts.

Forward pipeline:
  1. Read the canonical FineWeb-10B doc stream (``docs_selected.jsonl``
     produced by ``data/download_hf_docs_and_tokenize.py`` in the root repo).
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

Output layout — matches what ``train_gpt.py`` expects under
``DATA_DIR=./data`` with ``CASEOPS_ENABLED=1``:

    data/datasets/fineweb10B_sp8192_caseops/datasets/
      tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
      datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
        fineweb_train_000000.bin
        fineweb_train_000001.bin
        ...
        fineweb_val_000000.bin
        fineweb_val_bytes_000000.bin

Usage:

    python3 prepare_caseops_data.py \\
        --docs ./fineweb10B_raw/docs_selected.jsonl \\
        --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \\
        --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

Requirements: sentencepiece, numpy. CPU-only. Runs once; reused across seeds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

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


def _write_shard(out_path: pathlib.Path, arr: np.ndarray) -> None:
    """Write a uint16 shard in the standard header-prefixed format."""
    assert arr.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _iter_docs(docs_path: pathlib.Path):
    """Yield doc strings from a jsonl file (one json object per line)."""
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj


def _token_original_byte_counts(
    sp: spm.SentencePieceProcessor,
    original_text: str,
    transformed_text: str,
) -> np.ndarray:
    """Per-token canonical (pre-transform) UTF-8 byte counts."""
    proto = sp.encode_as_immutable_proto(transformed_text)
    byte_counts = surface_piece_original_byte_counts(
        (piece.surface for piece in proto.pieces),
        text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
    )
    return np.asarray(list(byte_counts), dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=pathlib.Path, help="Path to docs_selected.jsonl")
    ap.add_argument("--out",  required=True, type=pathlib.Path, help="Output datasets dir")
    ap.add_argument("--sp",   required=True, type=pathlib.Path, help="Path to CaseOps SP model")
    ap.add_argument("--val-docs", type=int, default=10_000, help="Validation docs count")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.sp))
    print(f"loaded sp: vocab={sp.vocab_size()}", flush=True)

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    val_buf_tokens: list[int] = []
    val_buf_bytes: list[int] = []
    train_buf: list[int] = []
    val_written = 0
    train_written = 0
    n_docs = 0

    for text in _iter_docs(args.docs):
        transformed = encode_lossless_caps_v2(text)
        token_ids = [BOS_ID] + sp.encode(transformed, out_type=int)
        if n_docs < args.val_docs:
            byte_counts = _token_original_byte_counts(sp, text, transformed)
            val_buf_tokens.extend(token_ids)
            val_buf_bytes.append(0)
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
    main()
