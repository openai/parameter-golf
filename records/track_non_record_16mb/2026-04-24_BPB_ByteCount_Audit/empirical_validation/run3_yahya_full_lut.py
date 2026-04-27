#!/usr/bin/env python3
"""Run 3 (corrected): Reconstruct yahya's exact LUT and compute ratios using
the same canonical/buggy formula as canonical_rescore.py.

Formula:
    canonical_total = sum(base_bytes[y]) + sum(has_leading_space[y] & ~is_boundary[x])
    buggy_total     = canonical_total + sum(has_leading_space[y] & ~is_boundary[x])
    ratio           = buggy_total / canonical_total

For yahya's LUT, his base_bytes already has the +1 baked in, so the formula
captures both his already-baked +1 and the eval-time +1 that doubles it.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sentencepiece as spm

TOKENIZER = "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
VAL = "/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
OUT = "/workspace/agent-pgolf/audit/empirical_validation/run3_yahya_full_lut.json"

SEQ_LEN = 2048
STRIDE = 64


def load_val():
    header = np.fromfile(VAL, dtype="<i4", count=256)
    assert int(header[0]) == 20240520
    n = int(header[2])
    return np.fromfile(VAL, dtype="<u2", count=n, offset=1024)


def build_yahya_lut(sp):
    """Yahya'\''s exact build_sentencepiece_luts (lines 206-219 of train_gdn_7k.py).

    For yahya, base_bytes[i] for leading-space pieces already contains
    len(stripped) + 1, the +1 being his bug. For byte tokens, no special
    branch -> base_bytes[i] = len(piece.encode('utf-8')) which is 6 for '<0xNN>'.
    """
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
    """Canonical PR #1727 LUT: base_bytes does NOT include the leading-space +1.

    Formula at scoring time:
      total = sum(base_bytes[y]) + sum(has_leading_space[y] & ~is_boundary[x])
    where base_bytes for leading-space tokens is len(piece.encode('utf-8')) - 1
    (i.e., the bytes excluding the leading space char, which is added back at eval).

    Actually reading the canonical PR #1727 more carefully, base_bytes for
    leading-space tokens stores len(stripped) and the eval adds +1. So:
      base_bytes[i] = len(stripped_piece.encode('utf-8')) for leading-space
      base_bytes[i] = len(piece.encode('utf-8'))         for non-leading-space
      base_bytes[i] = 1                                  for byte tokens
      base_bytes[i] = 0                                  for boundary tokens
    """
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


def compute_ratio_canonical_formula(val, base_bytes, has_space, is_boundary, mode):
    """Apply canonical_rescore.py'\''s formula:
      canonical_total = base_bytes[y].sum() + (has_space[y] & ~is_boundary[x]).sum()
      buggy_total = canonical_total + (has_space[y] & ~is_boundary[x]).sum()
    """
    n = val.shape[0]
    if mode == "sliding-window-boundary-masked":
        last_end = ((n - SEQ_LEN) // STRIDE) * STRIDE + SEQ_LEN
        if last_end > n:
            last_end = n
        y = val[1:last_end].astype(np.int64)
        x = val[:last_end - 1].astype(np.int64)
    else:
        y = val[1:].astype(np.int64)
        x = val[:-1].astype(np.int64)
    eval_add = int((has_space[y] & ~is_boundary[x]).sum())
    canonical_total = int(base_bytes[y].sum()) + eval_add
    buggy_total = canonical_total + eval_add
    return canonical_total, buggy_total, buggy_total / canonical_total, int(y.shape[0]), eval_add


def compute_ratio_yahya_formula(val, base_bytes, has_space, is_boundary, mode):
    """For yahya, his base_bytes already contains the +1. So eval'\''s formulation:
      his_canonical = base_bytes[y].sum()  (no eval-time +1, since LUT has it)
      his_buggy = base_bytes[y].sum() + (has_space[y] & ~is_boundary[x]).sum()
    """
    n = val.shape[0]
    if mode == "sliding-window-boundary-masked":
        last_end = ((n - SEQ_LEN) // STRIDE) * STRIDE + SEQ_LEN
        if last_end > n:
            last_end = n
        y = val[1:last_end].astype(np.int64)
        x = val[:last_end - 1].astype(np.int64)
    else:
        y = val[1:].astype(np.int64)
        x = val[:-1].astype(np.int64)
    eval_add = int((has_space[y] & ~is_boundary[x]).sum())
    canonical_total = int(base_bytes[y].sum())
    buggy_total = canonical_total + eval_add
    return canonical_total, buggy_total, buggy_total / canonical_total, int(y.shape[0]), eval_add


def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER)
    val = load_val()
    print(f"[run3] val: {val.shape[0]:,} tokens, vocab: {sp.GetPieceSize()}")
    print()

    yb, yh, yi = build_yahya_lut(sp)
    cb, ch, ci = build_canonical_lut(sp)

    print(f"[run3] yahya LUT base_bytes vocab sum: {int(yb.sum()):,}")
    print(f"[run3] canonical LUT base_bytes vocab sum: {int(cb.sum()):,}")
    print(f"[run3] vocab-level delta (yahya - canonical): {int(yb.sum()) - int(cb.sum()):,}")
    print()

    results = {}

    # Canonical, canonical formula (this should match canonical_rescore.py)
    for mode in ["sliding-window-boundary-masked", "all-tokens-boundary-masked"]:
        ct, bt, r, scored, ea = compute_ratio_canonical_formula(val, cb, ch, ci, mode)
        key = f"canonical_lut__canonical_formula__{mode}"
        results[key] = {"canonical_bytes": ct, "buggy_bytes": bt, "ratio": r,
                        "scored_tokens": scored, "eval_add": ea}
        print(f"[run3] {key}: ratio = {r:.10f}, c={ct:,}, b={bt:,}")

    print()

    # Yahya'\''s LUT, yahya formula (since his +1 is baked in)
    for mode in ["sliding-window-boundary-masked", "all-tokens-boundary-masked"]:
        ct, bt, r, scored, ea = compute_ratio_yahya_formula(val, yb, yh, yi, mode)
        key = f"yahya_lut__yahya_formula__{mode}"
        results[key] = {"canonical_bytes": ct, "buggy_bytes": bt, "ratio": r,
                        "scored_tokens": scored, "eval_add": ea}
        print(f"[run3] {key}: ratio = {r:.10f}, c={ct:,}, b={bt:,}")

    print()

    # Comparison summary
    yahya_quoted = 1.1746
    yahya_actual = results["yahya_lut__yahya_formula__sliding-window-boundary-masked"]["ratio"]
    canonical_actual = results["canonical_lut__canonical_formula__sliding-window-boundary-masked"]["ratio"]
    canonical_audit_tool = 1.1671413  # from run 1.5

    print(f"[run3] === COMPARISON ===")
    print(f"[run3] yahya quoted ratio:        {yahya_quoted}")
    print(f"[run3] yahya actual (his LUT):    {yahya_actual:.10f}")
    print(f"[run3]   diff:                    {yahya_actual - yahya_quoted:+.10f} ({(yahya_actual - yahya_quoted)/yahya_quoted*100:+.4f}%)")
    print(f"[run3] canonical (PR #1727 LUT):  {canonical_actual:.10f}")
    print(f"[run3]   audit tool reports:      {canonical_audit_tool}")
    print(f"[run3]   match:                   {abs(canonical_actual - canonical_audit_tool) < 1e-6}")

    output = {
        "ratios": results,
        "yahya_quoted_ratio": yahya_quoted,
        "yahya_actual_ratio_sliding_window": yahya_actual,
        "yahya_quoted_vs_actual_diff": yahya_actual - yahya_quoted,
        "yahya_quoted_vs_actual_pct_diff": (yahya_actual - yahya_quoted) / yahya_quoted * 100,
        "canonical_ratio_sliding_window": canonical_actual,
        "canonical_matches_audit_tool": abs(canonical_actual - canonical_audit_tool) < 1e-6,
        "tokenizer_path": TOKENIZER,
        "val_path": VAL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[run3] Wrote {OUT}")
    print(f"[run3] DONE")


if __name__ == "__main__":
    main()
