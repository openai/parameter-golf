#!/usr/bin/env python3
"""Run 2: empirically resolve yahya's byte-token classification.

Yahya's build_sentencepiece_luts (lines 206-219 of train_gdn_7k.py)
has no sp.is_byte branch. Byte tokens fall through to:
    base_bytes[i] = len(piece.encode("utf-8"))
For a byte piece "<0x00>", len("<0x00>".encode("utf-8")) == 6.
Canonical assigns 1.

This script computes the per-token byte assignment under both schemes,
counts byte tokens in val, and quantifies the contribution to inflation.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sentencepiece as spm

TOKENIZER = "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
VAL = "/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
OUT = "/workspace/agent-pgolf/audit/empirical_validation/run2_yahya_byte_token_check.json"


def load_val():
    header = np.fromfile(VAL, dtype="<i4", count=256)
    assert int(header[0]) == 20240520
    n = int(header[2])
    return np.fromfile(VAL, dtype="<u2", count=n, offset=1024)


def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER)
    vocab = sp.GetPieceSize()
    print(f"[run2] vocab: {vocab}")

    is_byte = np.zeros(vocab, dtype=bool)
    yahya_bytes = np.zeros(vocab, dtype=np.int64)
    canonical_bytes = np.zeros(vocab, dtype=np.int64)

    pieces = []
    for tid in range(vocab):
        is_byte[tid] = sp.IsByte(tid)
        piece = sp.IdToPiece(tid)
        # Yahya's logic for byte tokens: falls through to default branch
        yahya_bytes[tid] = len(piece.encode("utf-8"))
        # Canonical: byte tokens get 1
        if is_byte[tid]:
            canonical_bytes[tid] = 1
        else:
            canonical_bytes[tid] = len(piece.encode("utf-8"))
        if is_byte[tid] and len(pieces) < 10:
            pieces.append({"tid": tid, "piece": piece, "yahya": int(yahya_bytes[tid]), "canonical": int(canonical_bytes[tid])})

    n_byte_tokens = int(is_byte.sum())
    print(f"[run2] byte tokens in vocab: {n_byte_tokens}")
    print(f"[run2] sample byte token assignments:")
    for p in pieces:
        print(f"  tid={p['tid']:4d} piece={p['piece']!r:10s} yahya={p['yahya']} canonical={p['canonical']}")

    yahya_byte_total = int(yahya_bytes[is_byte].sum())
    canonical_byte_total = int(canonical_bytes[is_byte].sum())
    print(f"[run2] sum over byte tokens (per-vocab-id): yahya={yahya_byte_total} canonical={canonical_byte_total}")

    # Distribution of yahya's byte counts across byte tokens
    from collections import Counter
    yahya_dist = Counter(yahya_bytes[is_byte].tolist())
    canonical_dist = Counter(canonical_bytes[is_byte].tolist())
    print(f"[run2] yahya byte-count distribution (over byte tokens): {dict(yahya_dist)}")
    print(f"[run2] canonical byte-count distribution (over byte tokens): {dict(canonical_dist)}")

    # Now count byte tokens in val
    val = load_val()
    print(f"[run2] val tokens: {val.shape[0]:,}")

    val_byte_mask = is_byte[val]
    n_byte_in_val = int(val_byte_mask.sum())
    yahya_total_byte_bytes = int(yahya_bytes[val].sum() * val_byte_mask.astype(np.int64).sum() / max(1, n_byte_in_val)) if n_byte_in_val else 0

    # Better: actual contribution
    yahya_contribution = int(yahya_bytes[val][val_byte_mask].sum())
    canonical_contribution = int(canonical_bytes[val][val_byte_mask].sum())
    delta = yahya_contribution - canonical_contribution
    print(f"[run2] byte tokens in val: {n_byte_in_val:,}")
    print(f"[run2] yahya byte-token contribution to byte sum (val): {yahya_contribution:,}")
    print(f"[run2] canonical byte-token contribution to byte sum (val): {canonical_contribution:,}")
    print(f"[run2] delta (yahya - canonical): {delta:,}")

    bug_present = (yahya_byte_total != canonical_byte_total)
    if bug_present:
        verdict = "BUG_PRESENT"
        reasoning = ("Yahya's code lacks an sp.is_byte branch. Byte tokens fall through to "
                     "base_bytes[i] = len(piece.encode('utf-8')). For byte pieces of form "
                     "'<0xNN>', that gives 6 (or similar). Canonical assigns 1. "
                     f"Across {n_byte_tokens} byte tokens in vocab, yahya assigns "
                     f"{yahya_byte_total} bytes vs canonical {canonical_byte_total}. "
                     f"In val, byte tokens contribute {delta:,} extra bytes to the canonical numerator.")
    else:
        verdict = "BUG_ABSENT"
        reasoning = "Yahya's byte-token assignments match canonical. No bug."

    print(f"[run2] VERDICT: {verdict}")
    print(f"[run2] {reasoning}")

    output = {
        "n_byte_tokens_in_vocab": n_byte_tokens,
        "yahya_byte_count_distribution": {str(k): v for k, v in yahya_dist.items()},
        "canonical_byte_count_distribution": {str(k): v for k, v in canonical_dist.items()},
        "yahya_total_byte_token_bytes_vocab": yahya_byte_total,
        "canonical_total_byte_token_bytes_vocab": canonical_byte_total,
        "n_byte_tokens_in_val": n_byte_in_val,
        "yahya_byte_token_contribution_in_val": yahya_contribution,
        "canonical_byte_token_contribution_in_val": canonical_contribution,
        "delta_bytes_in_val": delta,
        "verdict": verdict,
        "verdict_reasoning": reasoning,
        "yahya_code_snippet": (
            "def build_sentencepiece_luts(sp, vocab_size, device):\n"
            "    base_bytes = torch.zeros(...)\n"
            "    for i in range(vocab_size):\n"
            "        piece = sp.id_to_piece(i)\n"
            "        raw = piece.encode('utf-8')\n"
            "        base_bytes[i] = len(raw)            # NO sp.is_byte branch -> bytes get utf-8 length of literal\n"
            "        if piece.startswith('\u2581'):\n"
            "            has_space[i] = True\n"
            "            base_bytes[i] = len(piece[1:].encode('utf-8')) + 1   # +1 bug\n"
            "        if sp.is_control(i) or sp.is_unknown(i):\n"
            "            is_boundary[i] = True            # missing sp.is_unused\n"
        ),
        "tokenizer_path": TOKENIZER,
        "val_path": VAL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[run2] Wrote {OUT}")
    print(f"[run2] DONE")


if __name__ == "__main__":
    main()
