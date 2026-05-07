"""Parallel version of ``prepare_caseops_data.py``.

Same output format / semantics as the single-process script, but uses a
``multiprocessing.Pool`` to parallelize the CPU-heavy work (CaseOps
transform, SentencePiece tokenization, per-token original byte counts).

Design:
  - Main process: reads ``docs_selected.jsonl`` sequentially, streams docs
    into ``pool.imap(..., chunksize=...)`` so worker output order matches
    input order (shard contents stay byte-identical to the serial script).
  - Each worker: lazy-loads the SentencePiece model once via ``initializer``,
    then for each doc returns ``(token_ids_uint16, byte_counts_uint16 | None)``.
    Validation docs also return the per-token original UTF-8 byte sidecar.
  - Main process accumulates into the train / val buffers and flushes full
    10M-token shards, exactly like the serial version.

Usage (drop-in replacement, plus ``-j`` / ``--jobs``):

    python3 prepare_caseops_data_mp.py \\
        --docs ./fineweb10B_raw/docs_selected.jsonl \\
        --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \\
        --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \\
        --jobs 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from typing import Iterator

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
SHARD_TOKENS = 10**8  # tokens per shard — MUST match prepare_caseops_data.py
BOS_ID = 1  # SP model's <s> control token; train_gpt.py:_find_docs requires BOS per doc


# ---------------------------------------------------------------------------
# Shard I/O (identical to the serial script)
# ---------------------------------------------------------------------------

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


def _iter_docs(docs_path: pathlib.Path) -> Iterator[str]:
    """Yield doc strings from a jsonl file (one json object per line)."""
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj


# ---------------------------------------------------------------------------
# Worker-side: lazy SP load, per-doc transform + tokenize + byte sidecar
# ---------------------------------------------------------------------------

# Module-level globals populated by ``_worker_init`` in each child process.
_WORKER_SP: spm.SentencePieceProcessor | None = None
_WORKER_SP_PATH: str | None = None


def _worker_init(sp_path: str) -> None:
    """Load the SentencePiece model exactly once per worker process."""
    global _WORKER_SP, _WORKER_SP_PATH
    _WORKER_SP = spm.SentencePieceProcessor(model_file=sp_path)
    _WORKER_SP_PATH = sp_path


def _process_doc(payload: tuple[str, bool]) -> tuple[bytes, bytes | None]:
    """Transform + tokenize one doc.

    Returns raw uint16 byte buffers (``tobytes()``) so that pickling across
    the process boundary is as cheap as possible — a bytes object is much
    cheaper to pickle than a Python list of ints or a numpy array.

    Args:
        payload: ``(text, need_bytes)`` — ``need_bytes`` is True only for
            validation docs (we emit the per-token original-bytes sidecar
            for those).

    Returns:
        ``(tokens_bytes, bytes_sidecar_bytes_or_None)`` where both buffers
        are little-endian uint16 arrays. For non-val docs the second item
        is ``None`` to save IPC bandwidth.
    """
    text, need_bytes = payload
    assert _WORKER_SP is not None, "worker SP model not initialized"
    sp = _WORKER_SP

    transformed = encode_lossless_caps_v2(text)

    if need_bytes:
        # encode_as_immutable_proto gives us both ids and surfaces in one pass.
        proto = sp.encode_as_immutable_proto(transformed)
        token_ids_arr = np.empty(len(proto.pieces) + 1, dtype=np.uint16)
        token_ids_arr[0] = BOS_ID
        for idx, piece in enumerate(proto.pieces, start=1):
            token_ids_arr[idx] = piece.id

        byte_counts = surface_piece_original_byte_counts(
            (piece.surface for piece in proto.pieces),
            text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
        )
        bytes_arr = np.empty(len(byte_counts) + 1, dtype=np.uint16)
        bytes_arr[0] = 0  # BOS contributes 0 original bytes
        bytes_arr[1:] = np.asarray(byte_counts, dtype=np.uint16)
        return token_ids_arr.tobytes(), bytes_arr.tobytes()

    # Train doc — only need ids, skip the proto overhead.
    ids = sp.encode(transformed, out_type=int)
    token_ids_arr = np.empty(len(ids) + 1, dtype=np.uint16)
    token_ids_arr[0] = BOS_ID
    token_ids_arr[1:] = np.asarray(ids, dtype=np.uint16)
    return token_ids_arr.tobytes(), None


# ---------------------------------------------------------------------------
# Main-process: shard buffers + flush
# ---------------------------------------------------------------------------

class ShardBuffer:
    """Accumulates uint16 byte buffers and flushes 10M-token shards."""

    def __init__(self, out_dir: pathlib.Path, prefix: str, has_sidecar: bool) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.has_sidecar = has_sidecar
        # Keep raw bytes buffers — we'll np.frombuffer them at flush time.
        self._tok_chunks: list[bytes] = []
        self._byte_chunks: list[bytes] = []
        self._tok_tokens = 0  # running sum of tokens in _tok_chunks
        self._shards_written = 0

    def add(self, tok_buf: bytes, byte_buf: bytes | None) -> None:
        self._tok_chunks.append(tok_buf)
        # uint16 => 2 bytes per token
        self._tok_tokens += len(tok_buf) // 2
        if self.has_sidecar:
            assert byte_buf is not None
            self._byte_chunks.append(byte_buf)
        while self._tok_tokens >= SHARD_TOKENS:
            self._flush_full()

    def _concat_uint16(self, chunks: list[bytes]) -> np.ndarray:
        """Concatenate many bytes chunks into a single uint16 array."""
        if len(chunks) == 1:
            return np.frombuffer(chunks[0], dtype=np.uint16).copy()
        return np.frombuffer(b"".join(chunks), dtype=np.uint16).copy()

    def _flush_full(self) -> None:
        """Emit exactly one full SHARD_TOKENS-sized shard, keep the remainder."""
        tok_all = self._concat_uint16(self._tok_chunks)
        shard_tokens = tok_all[:SHARD_TOKENS]
        remainder_tokens = tok_all[SHARD_TOKENS:]

        _write_shard(
            self.out_dir / f"fineweb_{self.prefix}_{self._shards_written:06d}.bin",
            shard_tokens,
        )

        if self.has_sidecar:
            byte_all = self._concat_uint16(self._byte_chunks)
            assert byte_all.size == tok_all.size, (
                f"token/byte sidecar mismatch: {tok_all.size} vs {byte_all.size}"
            )
            shard_bytes = byte_all[:SHARD_TOKENS]
            remainder_bytes = byte_all[SHARD_TOKENS:]
            _write_shard(
                self.out_dir / f"fineweb_{self.prefix}_bytes_{self._shards_written:06d}.bin",
                shard_bytes,
            )
            # Reset chunks to a single remainder chunk.
            self._byte_chunks = (
                [remainder_bytes.tobytes()] if remainder_bytes.size else []
            )

        self._tok_chunks = [remainder_tokens.tobytes()] if remainder_tokens.size else []
        self._tok_tokens = int(remainder_tokens.size)
        self._shards_written += 1

    def flush_tail(self) -> None:
        """Write any residual (<SHARD_TOKENS) tail as a final short shard."""
        if self._tok_tokens == 0:
            return
        tok_all = self._concat_uint16(self._tok_chunks)
        _write_shard(
            self.out_dir / f"fineweb_{self.prefix}_{self._shards_written:06d}.bin",
            tok_all,
        )
        if self.has_sidecar:
            byte_all = self._concat_uint16(self._byte_chunks)
            assert byte_all.size == tok_all.size
            _write_shard(
                self.out_dir / f"fineweb_{self.prefix}_bytes_{self._shards_written:06d}.bin",
                byte_all,
            )
        self._shards_written += 1
        self._tok_chunks = []
        self._byte_chunks = []
        self._tok_tokens = 0

    @property
    def shards_written(self) -> int:
        return self._shards_written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _iter_payloads(
    docs_path: pathlib.Path, val_docs: int
) -> Iterator[tuple[str, bool]]:
    """Yield ``(text, need_bytes)`` in docs_selected.jsonl order."""
    for doc_idx, text in enumerate(_iter_docs(docs_path)):
        yield text, doc_idx < val_docs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--docs", required=True, type=pathlib.Path, help="Path to docs_selected.jsonl")
    ap.add_argument("--out",  required=True, type=pathlib.Path, help="Output datasets dir")
    ap.add_argument("--sp",   required=True, type=pathlib.Path, help="Path to CaseOps SP model")
    ap.add_argument("--val-docs", type=int, default=50_000, help="Validation docs count")
    ap.add_argument(
        "-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes (default: cpu_count-1)",
    )
    ap.add_argument(
        "--chunksize", type=int, default=32,
        help="Pool.imap chunksize — docs per IPC round-trip (default: 32)",
    )
    args = ap.parse_args()

    # Sanity-load SP in the main process just to print vocab size / fail fast.
    sp_probe = spm.SentencePieceProcessor(model_file=str(args.sp))
    print(f"loaded sp: vocab={sp_probe.vocab_size()}  jobs={args.jobs}  chunksize={args.chunksize}", flush=True)
    del sp_probe

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    val_buf = ShardBuffer(train_out, prefix="val", has_sidecar=True)
    train_buf = ShardBuffer(train_out, prefix="train", has_sidecar=False)

    t0 = time.time()
    n_docs = 0

    # ``spawn`` is safer with sentencepiece + OS threads; avoids any pre-loaded
    # state in the parent from being inherited by fork (which can deadlock).
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.jobs,
        initializer=_worker_init,
        initargs=(str(args.sp),),
    ) as pool:
        payloads = _iter_payloads(args.docs, args.val_docs)
        # ``imap`` preserves input order — critical for reproducible shards.
        for doc_idx, (tok_buf, byte_buf) in enumerate(
            pool.imap(_process_doc, payloads, chunksize=args.chunksize)
        ):
            is_val = doc_idx < args.val_docs
            if is_val:
                val_buf.add(tok_buf, byte_buf)
            else:
                train_buf.add(tok_buf, None)
            n_docs = doc_idx + 1
            if n_docs % 10_000 == 0:
                elapsed = time.time() - t0
                rate = n_docs / elapsed if elapsed > 0 else 0.0
                print(
                    f"  processed {n_docs} docs  "
                    f"train_shards={train_buf.shards_written}  "
                    f"val_shards={val_buf.shards_written}  "
                    f"rate={rate:.1f} docs/s  elapsed={elapsed:.1f}s",
                    flush=True,
                )

    # Flush tail buffers into final (possibly short) shards.
    val_buf.flush_tail()
    train_buf.flush_tail()

    elapsed = time.time() - t0
    print(
        f"done. docs={n_docs} "
        f"train_shards={train_buf.shards_written} "
        f"val_shards={val_buf.shards_written} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
