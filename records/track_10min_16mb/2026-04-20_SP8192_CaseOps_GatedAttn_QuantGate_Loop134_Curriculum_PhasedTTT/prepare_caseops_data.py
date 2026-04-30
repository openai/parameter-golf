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

This script is intended to reproduce the actual shard format used by the
original CaseOps export path from PR #1729 / the HF-hosted dataset:

- every document is prepended with ``bos_id``
- validation byte sidecars include a matching leading ``0`` byte count
- the default validation split is the canonical 50,000-doc challenge split

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
SHARD_TOKENS = 100_000_000  # tokens per shard — matches the original CaseOps export path


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
            # Support both {"text": ...} and raw strings.
            yield obj["text"] if isinstance(obj, dict) else obj


def _encode_with_original_byte_counts(
    sp: spm.SentencePieceProcessor, text: str
) -> tuple[np.ndarray, np.ndarray]:
    """Match the original CaseOps exporter exactly.

    The original PR #1729 export path tokenized via
    ``encode_as_immutable_proto`` and computed canonical byte counts from the
    exact piece surfaces using ``surface_piece_original_byte_counts``. Reuse
    that logic here so the rebuilt validation sidecar matches the true
    CaseOps dataset format byte-for-byte.
    """
    transformed = encode_lossless_caps_v2(text)
    proto = sp.encode_as_immutable_proto(transformed)
    token_ids = np.fromiter((piece.id for piece in proto.pieces), dtype=np.int32)
    byte_counts = np.asarray(
        surface_piece_original_byte_counts(
            (piece.surface for piece in proto.pieces),
            text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
        ),
        dtype=np.uint16,
    )
    if token_ids.shape[0] != byte_counts.shape[0]:
        raise ValueError("token id count and byte count length disagree")
    return token_ids, byte_counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=pathlib.Path, help="Path to docs_selected.jsonl")
    ap.add_argument("--out",  required=True, type=pathlib.Path, help="Output datasets dir")
    ap.add_argument("--sp",   required=True, type=pathlib.Path, help="Path to CaseOps SP model")
    ap.add_argument("--val-docs", type=int, default=50_000, help="Validation docs count")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.sp))
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("tokenizer must define a valid bos_id")
    print(f"loaded sp: vocab={sp.vocab_size()} bos_id={bos_id}", flush=True)

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    val_buf_tokens: list[int] = []
    val_buf_bytes: list[int] = []
    train_buf: list[int] = []
    val_written = 0
    train_written = 0
    n_docs = 0

    for text in _iter_docs(args.docs):
        piece_ids, piece_byte_counts = _encode_with_original_byte_counts(sp, text)
        token_ids = np.empty(piece_ids.size + 1, dtype=np.int32)
        token_ids[0] = bos_id
        token_ids[1:] = piece_ids
        if n_docs < args.val_docs:
            # Validation doc — also compute byte sidecar
            if piece_byte_counts.shape[0] != piece_ids.shape[0]:
                raise ValueError("token id count and original byte count length disagree")
            byte_counts = np.zeros(token_ids.shape[0], dtype=np.int32)
            byte_counts[1:] = piece_byte_counts.astype(np.int32, copy=False)
            val_buf_tokens.extend(int(t) for t in token_ids)
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
            train_buf.extend(int(t) for t in token_ids)
            if len(train_buf) >= SHARD_TOKENS:
                _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                             np.array(train_buf[:SHARD_TOKENS], dtype=np.uint16))
                train_buf = train_buf[SHARD_TOKENS:]
                train_written += 1
        n_docs += 1
        if n_docs % 10_000 == 0:
            print(f"  processed {n_docs} docs  train_shards={train_written}  val_shards={val_written}", flush=True)

    # Flush tail buffers into final (possibly short) shards.
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
