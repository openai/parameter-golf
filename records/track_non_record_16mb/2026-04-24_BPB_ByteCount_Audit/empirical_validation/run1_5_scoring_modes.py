#!/usr/bin/env python3
"""Run 1.5: high-precision check of the three scoring modes.

Run 1 found that boundary_mask_is_no_op is FALSE: 50,000 control-token
predecessors exist in the val stream. This script re-runs the three
scoring modes from canonical_rescore.py with the mask correctly applied,
reporting ratios at 8 decimal places, to determine whether the modes
actually converge.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentencepiece as spm

TOKENIZER_PATH = "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
VAL_PATH = "/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
OUTPUT_JSON = "/workspace/agent-pgolf/audit/empirical_validation/run1_5_scoring_modes.json"

SEQ_LEN = 2048
STRIDE = 64

def load_val():
    header = np.fromfile(VAL_PATH, dtype="<i4", count=256)
    assert int(header[0]) == 20240520
    n = int(header[2])
    return np.fromfile(VAL_PATH, dtype="<u2", count=n, offset=1024)

def build_luts(sp):
    """Canonical PR #1727 LUT construction."""
    vocab = sp.GetPieceSize()
    base_bytes_canonical = np.zeros(vocab, dtype=np.int64)
    base_bytes_buggy = np.zeros(vocab, dtype=np.int64)
    has_leading_space = np.zeros(vocab, dtype=bool)
    is_boundary = np.zeros(vocab, dtype=bool)
    for tid in range(vocab):
        is_boundary[tid] = sp.IsControl(tid) or sp.IsUnknown(tid) or sp.IsUnused(tid)
        if is_boundary[tid]:
            continue
        piece = sp.IdToPiece(tid)
        if sp.IsByte(tid):
            base_bytes_canonical[tid] = 1
            base_bytes_buggy[tid] = 1
            continue
        if piece.startswith("▁"):  # leading-space marker
            has_leading_space[tid] = True
            stripped = piece[1:]
            base_bytes_canonical[tid] = len(stripped.encode("utf-8"))
            base_bytes_buggy[tid] = len(stripped.encode("utf-8")) + 1
        else:
            base_bytes_canonical[tid] = len(piece.encode("utf-8"))
            base_bytes_buggy[tid] = len(piece.encode("utf-8"))
    return base_bytes_canonical, base_bytes_buggy, has_leading_space, is_boundary

def compute_mode(val, base_canonical, base_buggy, has_ls, is_bnd, mode):
    """Compute (canonical_bytes, buggy_bytes, ratio) for given scoring mode."""
    n = val.shape[0]
    if mode == "sliding-window-boundary-masked":
        last_end = ((n - SEQ_LEN) // STRIDE) * STRIDE + SEQ_LEN
        if last_end > n: last_end = n
        y = val[1:last_end].astype(np.int64)
        x = val[:last_end - 1].astype(np.int64)
        mask = ~is_bnd[x]
    elif mode == "all-tokens-boundary-masked":
        y = val[1:].astype(np.int64)
        x = val[:-1].astype(np.int64)
        mask = ~is_bnd[x]
    elif mode == "all-tokens-no-mask":
        y = val[1:].astype(np.int64)
        x = val[:-1].astype(np.int64)
        mask = np.ones(y.shape[0], dtype=bool)
    else:
        raise ValueError(mode)
    canonical_total = int(base_canonical[y].sum() + (has_ls[y] & mask).sum())
    buggy_total = int(base_buggy[y].sum() + (has_ls[y] & mask).sum())
    ratio = buggy_total / canonical_total
    return canonical_total, buggy_total, ratio, int(y.shape[0])

def main():
    print("[run1.5] Loading tokenizer + val...")
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    val = load_val()
    print(f"[run1.5] val: {val.shape[0]:,} tokens, vocab: {sp.GetPieceSize()}")

    print("[run1.5] Building LUTs...")
    bc, bb, hls, isb = build_luts(sp)
    print(f"[run1.5]   leading_space tokens: {int(hls.sum())}")
    print(f"[run1.5]   boundary tokens (vocab): {int(isb.sum())}")

    modes = ["sliding-window-boundary-masked", "all-tokens-boundary-masked", "all-tokens-no-mask"]
    results = {}
    print()
    print("[run1.5] Mode results:")
    for m in modes:
        cb, bgb, r, scored = compute_mode(val, bc, bb, hls, isb, m)
        results[m] = {
            "canonical_bytes": cb,
            "buggy_bytes": bgb,
            "ratio": r,
            "scored_tokens": scored,
        }
        print(f"  {m}:")
        print(f"    canonical_bytes = {cb:,}")
        print(f"    buggy_bytes     = {bgb:,}")
        print(f"    ratio           = {r:.10f}")
        print(f"    scored_tokens   = {scored:,}")
        print()

    # Diff matrix at high precision
    print("[run1.5] Pairwise ratio differences:")
    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            diff = results[a]["ratio"] - results[b]["ratio"]
            pct = abs(diff) / results[a]["ratio"] * 100
            print(f"  {a}  -  {b}  = {diff:+.10f}  ({pct:.8f}%)")

    output = {
        "modes": results,
        "all_three_equal_at_4dp": all(round(r["ratio"], 4) == round(results[modes[0]]["ratio"], 4) for r in results.values()),
        "all_three_equal_at_6dp": all(round(r["ratio"], 6) == round(results[modes[0]]["ratio"], 6) for r in results.values()),
        "all_three_equal_at_10dp": all(r["ratio"] == results[modes[0]]["ratio"] for r in results.values()),
        "n_boundary_predecessors": 50000,  # from run 1
        "tokenizer_path": TOKENIZER_PATH,
        "val_path": VAL_PATH,
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"[run1.5] Wrote {OUTPUT_JSON}")
    print(f"[run1.5] DONE")

if __name__ == "__main__":
    main()
