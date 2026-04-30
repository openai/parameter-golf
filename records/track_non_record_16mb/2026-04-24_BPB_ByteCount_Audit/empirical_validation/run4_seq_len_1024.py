#!/usr/bin/env python3
"""Run 4: Test whether yahya's 1.1746 reproduces with seq_len=1024.

Yahya's train_gdn_7k.py defaults to eval_seq_len=1024 (line 69), and the
audit's reproduction used 2048. His submission.json confirms SP8192 tokenizer.
The hypothesis: yahya's PR #1734 disclosure analysis ran with seq_len=1024
on SP8192 val, producing 1.1746 instead of our 1.1655 (which is seq_len=2048).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sentencepiece as spm

TOKENIZER = "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
VAL = "/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
OUT = "/workspace/agent-pgolf/audit/empirical_validation/run4_seq_len_1024.json"


def load_val():
    header = np.fromfile(VAL, dtype="<i4", count=256)
    assert int(header[0]) == 20240520
    n = int(header[2])
    return np.fromfile(VAL, dtype="<u2", count=n, offset=1024)


def build_yahya_lut(sp):
    """Yahya's exact build_sentencepiece_luts (lines 206-219)."""
    vocab = sp.GetPieceSize()
    base_bytes = np.zeros(vocab, dtype=np.int64)
    has_space = np.zeros(vocab, dtype=bool)
    is_boundary = np.zeros(vocab, dtype=bool)
    for i in range(vocab):
        piece = sp.IdToPiece(i)
        raw = piece.encode("utf-8")
        base_bytes[i] = len(raw)
        if piece.startswith("\u2581"):
            has_space[i] = True
            base_bytes[i] = len(piece[1:].encode("utf-8")) + 1
        if sp.IsControl(i) or sp.IsUnknown(i):
            is_boundary[i] = True
    return base_bytes, has_space, is_boundary


def build_canonical_lut(sp):
    """Canonical PR #1727 LUT."""
    vocab = sp.GetPieceSize()
    base_bytes = np.zeros(vocab, dtype=np.int64)
    has_space = np.zeros(vocab, dtype=bool)
    is_boundary = np.zeros(vocab, dtype=bool)
    for i in range(vocab):
        is_boundary[i] = sp.IsControl(i) or sp.IsUnknown(i) or sp.IsUnused(i)
        if is_boundary[i]:
            continue
        piece = sp.IdToPiece(i)
        if sp.IsByte(i):
            base_bytes[i] = 1
            continue
        if piece.startswith("\u2581"):
            has_space[i] = True
            stripped = piece[1:]
            base_bytes[i] = len(stripped.encode("utf-8"))
        else:
            base_bytes[i] = len(piece.encode("utf-8"))
    return base_bytes, has_space, is_boundary


def compute_ratio_yahya_eval(val, base_bytes, has_space, is_boundary, seq_len, stride):
    """Replicate yahya's eval_val_sliding scoring exactly.

    Yahya's per-window scored region: s = 0 if ws==0 else max(wlen-stride, 0)
    Token count: wlen - s tokens scored per window.
    Byte count: base_bytes[tgt] + (has_space[tgt] & ~is_boundary[prev])
    """
    n = val.shape[0]
    total_tokens = n - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    canonical_total = 0
    eval_add_total = 0
    scored_tokens = 0

    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        # Tokens scored in this window: positions s..wlen within the window
        # tgt = val[ws+1+s : ws+1+wlen]
        # prev = val[ws+s : ws+wlen]
        if s >= wlen:
            continue
        tgt = val[ws + 1 + s : ws + 1 + wlen].astype(np.int64)
        prev = val[ws + s : ws + wlen].astype(np.int64)

        canonical_total += int(base_bytes[tgt].sum())
        eval_add_total += int((has_space[tgt] & ~is_boundary[prev]).sum())
        scored_tokens += int(tgt.shape[0])

    return canonical_total, eval_add_total, scored_tokens


def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER)
    val = load_val()
    print(f"[run4] val: {val.shape[0]:,} tokens")
    print()

    yb, yh, yi = build_yahya_lut(sp)
    cb, ch, ci = build_canonical_lut(sp)

    results = {}
    yahya_quoted = 1.1746

    for seq_len, stride in [(2048, 64), (1024, 64), (1024, 1024)]:
        print(f"[run4] === seq_len={seq_len}, stride={stride} ===")

        # Canonical LUT, canonical formula
        c_total, c_eval_add, c_scored = compute_ratio_yahya_eval(
            val, cb, ch, ci, seq_len, stride
        )
        c_canonical = c_total + c_eval_add
        c_buggy = c_canonical + c_eval_add
        c_ratio = c_buggy / c_canonical
        print(f"  canonical: ratio={c_ratio:.10f}, scored={c_scored:,}")

        # Yahya's LUT, yahya formula (his +1 baked in)
        y_total, y_eval_add, y_scored = compute_ratio_yahya_eval(
            val, yb, yh, yi, seq_len, stride
        )
        y_canonical = y_total
        y_buggy = y_canonical + y_eval_add
        y_ratio = y_buggy / y_canonical
        print(f"  yahya:     ratio={y_ratio:.10f}, scored={y_scored:,}")

        diff = y_ratio - yahya_quoted
        print(f"  yahya - quoted ({yahya_quoted}): {diff:+.6f} ({diff/yahya_quoted*100:+.4f}%)")
        print()

        results[f"seq_len_{seq_len}_stride_{stride}"] = {
            "canonical_lut_ratio": c_ratio,
            "yahya_lut_ratio": y_ratio,
            "scored_tokens": y_scored,
            "yahya_minus_quoted_pct": (y_ratio - yahya_quoted) / yahya_quoted * 100,
        }

    output = {
        "yahya_quoted_ratio": yahya_quoted,
        "results": results,
        "tokenizer_path": TOKENIZER,
        "val_path": VAL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[run4] Wrote {OUT}")


if __name__ == "__main__":
    main()
