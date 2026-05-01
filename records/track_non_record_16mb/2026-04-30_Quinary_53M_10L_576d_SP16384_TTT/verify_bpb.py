"""Standalone BPB-denominator correctness check.

Verifies that train_gpt.py's per-token byte-count LUT matches independent
SentencePiece decoding on the *same token stream and same scored slice* used by
training/evaluation.

Run from the submission directory after setup.sh:

    python3 verify_bpb.py

Optional env vars:
    TOKENIZER_PATH, DATA_PATH, VOCAB_SIZE, TRAIN_SEQ_LEN
"""

from __future__ import annotations

import glob
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm


SP_PATH = os.environ.get(
    "TOKENIZER_PATH", "./data/canonical/tokenizers/fineweb_16384_bpe.model"
)
DATA_PATH = os.environ.get(
    "DATA_PATH", "./data/canonical/datasets/fineweb10B_sp16384"
)
VAL_GLOB = os.environ.get("VAL_GLOB", str(Path(DATA_PATH) / "fineweb_val_*.bin"))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "16384"))
TRAIN_SEQ_LEN = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype("<i4").itemsize


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ld_shard(path: Path) -> np.ndarray:
    """Same shard loader as train_gpt.py: 256-int32 header, then u16 tokens."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise ValueError(f"{path}: short header ({header.size} int32s)")
    if int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise ValueError(
            f"{path}: bad header magic/version "
            f"({int(header[0])}, {int(header[1])})"
        )
    ntok = int(header[2])
    expected_min_size = HEADER_BYTES + ntok * np.dtype("<u2").itemsize
    actual_size = path.stat().st_size
    if actual_size < expected_min_size:
        raise ValueError(
            f"{path}: file too small for ntok={ntok}: "
            f"size={actual_size}, expected_at_least={expected_min_size}"
        )
    return np.fromfile(path, dtype="<u2", offset=HEADER_BYTES, count=ntok).astype(
        np.int64
    )


def build_luts(sp: spm.SentencePieceProcessor, vocab_size: int):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int32)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_token[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary_token


def lut_byte_count(
    tokens: np.ndarray,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: np.ndarray,
) -> int:
    """Sum eval bytes over tokens[1:], with tokens[:-1] as previous context."""
    tgt = tokens[1:]
    prev = tokens[:-1]
    base = base_bytes[tgt].astype(np.int64)
    extra = (has_leading_space[tgt] & ~is_boundary_token[prev]).astype(np.int64)
    return int((base + extra).sum())


def bos_positions(tokens: np.ndarray, bos_id: int) -> np.ndarray:
    return np.flatnonzero(tokens == bos_id)


def decoded_bytes_docwise(
    tokens: np.ndarray,
    sp: spm.SentencePieceProcessor,
    bos_id: int,
    is_boundary_token: np.ndarray,
) -> int:
    """Decode bytes for tokens[1:] using BOS as an explicit document reset.

    This mirrors the LUT convention: BOS/control tokens emit zero bytes, and the
    first text token after BOS does not pay an artificial leading-space byte.

    The normal canonical validation stream should start with BOS. If it does not,
    we fall back to a prefix-subtraction check for the initial partial segment.
    """
    if len(tokens) < 2:
        return 0

    bpos = bos_positions(tokens, bos_id)
    total = 0

    # Initial partial segment before the first BOS in this slice. This should not
    # happen for the exact eval slice; if it does, compare by prefix subtraction.
    first_bos = int(bpos[0]) if len(bpos) else len(tokens)
    if first_bos > 0:
        initial = tokens[:first_bos]
        full = sp.decode(initial.tolist()).encode("utf-8")
        prefix = sp.decode(initial[:1].tolist()).encode("utf-8")
        total += len(full) - len(prefix)

    # BOS-delimited document segments. For a BOS at position s, target bytes start
    # at s+1 and continue until the next BOS or end of slice.
    for i, s0 in enumerate(bpos):
        s = int(s0)
        e = int(bpos[i + 1]) if i + 1 < len(bpos) else len(tokens)
        if e > s + 1:
            total += len(sp.decode(tokens[s + 1 : e].tolist()).encode("utf-8"))
        # If e == s+1, the document is empty or the slice ends at BOS; emits 0.
    return total


def check_slice(
    label: str,
    tokens: np.ndarray,
    sp: spm.SentencePieceProcessor,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: np.ndarray,
    bos_id: int,
) -> bool:
    lut_sum = lut_byte_count(tokens, base_bytes, has_leading_space, is_boundary_token)
    decoded_bytes = decoded_bytes_docwise(tokens, sp, bos_id, is_boundary_token)
    delta = lut_sum - decoded_bytes
    status = "PASS" if delta == 0 else "FAIL"
    starts_boundary = bool(is_boundary_token[int(tokens[0])])
    print(
        f"  {status}  {label:36s} targets={len(tokens)-1:>10,d} "
        f"lut_bytes={lut_sum:>12,d} decoded_bytes={decoded_bytes:>12,d} "
        f"delta={delta:+d} start_boundary={starts_boundary}"
    )
    return delta == 0


def main() -> int:
    print("=" * 78)
    print("BPB byte-count LUT verification")
    print("=" * 78)

    sp_path = Path(SP_PATH)
    if not sp_path.exists():
        print(f"Tokenizer not found at {sp_path}; run setup.sh first.")
        return 2
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    print(f"tokenizer    : {sp_path} (vocab={sp.vocab_size()})")
    print(f"tokenizer sha: {sha256_file(sp_path)}")

    bos = sp.bos_id()
    eos = sp.eos_id()
    unk = sp.unk_id()
    print(f"BOS id={bos} ({sp.id_to_piece(bos)!r})  EOS id={eos} ({sp.id_to_piece(eos)!r})  UNK id={unk}")

    val_files = sorted(glob.glob(VAL_GLOB))
    if not val_files:
        print(f"No val shards matched {VAL_GLOB}")
        return 2
    print(f"val shards   : {len(val_files)} ({val_files[0]} ... )")

    base_bytes, has_leading_space, is_boundary_token = build_luts(sp, VOCAB_SIZE)

    n_byte = sum(1 for tid in range(sp.vocab_size()) if sp.is_byte(tid))
    n_control = sum(
        1
        for tid in range(sp.vocab_size())
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid)
    )
    n_lead = int(has_leading_space[: sp.vocab_size()].sum())
    n_bound = int(is_boundary_token[: sp.vocab_size()].sum())
    print(
        f"LUT stats    : byte-fallback={n_byte}  control/unknown/unused={n_control}  "
        f"with-leading-space={n_lead}  boundary={n_bound}"
    )

    tokens_full = np.concatenate([ld_shard(Path(f)) for f in val_files])
    print(f"shard tokens : {tokens_full.size:,}")
    if tokens_full.size < 2:
        print("Too few tokens")
        return 2

    bpos = bos_positions(tokens_full, bos)
    print(
        f"BOS positions: {len(bpos):,}; first={int(bpos[0]) if len(bpos) else 'NONE'} "
        f"(expect 50,000 docs and normally first=0)"
    )
    unk_count = int((tokens_full == unk).sum()) if unk >= 0 else 0
    print(f"UNK count    : {unk_count}")

    all_pass = True
    print("\nPer-slice LUT-vs-decode parity:")

    # Exact scored slice used by train_gpt.py: truncate to a TRAIN_SEQ_LEN target multiple.
    u = ((tokens_full.size - 1) // TRAIN_SEQ_LEN) * TRAIN_SEQ_LEN
    tokens_eval = tokens_full[: u + 1]
    omitted_targets = (tokens_full.size - 1) - u
    print(
        f"eval slice   : train_seq_len={TRAIN_SEQ_LEN} target_count={u:,} "
        f"omitted_tail_targets={omitted_targets:,}"
    )
    all_pass &= check_slice(
        "exact eval slice",
        tokens_eval,
        sp,
        base_bytes,
        has_leading_space,
        is_boundary_token,
        bos,
    )

    # Whole untruncated stream.
    all_pass &= check_slice(
        "full untruncated stream",
        tokens_full,
        sp,
        base_bytes,
        has_leading_space,
        is_boundary_token,
        bos,
    )

    # BOS-delimited prefix document checks.
    if len(bpos) >= 2:
        all_pass &= check_slice(
            "doc at first BOS",
            tokens_full[int(bpos[0]) : int(bpos[1])],
            sp,
            base_bytes,
            has_leading_space,
            is_boundary_token,
            bos,
        )
    for n_docs in (10, 100, 1000):
        if len(bpos) >= n_docs + 1:
            all_pass &= check_slice(
                f"first {n_docs} BOS docs",
                tokens_full[int(bpos[0]) : int(bpos[n_docs])],
                sp,
                base_bytes,
                has_leading_space,
                is_boundary_token,
                bos,
            )

    print()
    if all_pass and unk_count == 0:
        print("ALL CHECKS PASS — LUT bytes match SentencePiece decoder bytes on the eval slice.")
        return 0
    if unk_count != 0:
        print("FAIL — validation contains UNK tokens; zero-byte UNK accounting is unsafe.")
    else:
        print("AT LEAST ONE CHECK FAILED — investigate before trusting BPB scores.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
