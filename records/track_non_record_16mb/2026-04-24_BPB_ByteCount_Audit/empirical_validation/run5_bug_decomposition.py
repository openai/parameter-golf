#!/usr/bin/env python3
"""Run 5: Decompose the inflation ratio into per-bug-family contributions.

Yahya's LUT has three deviations from canonical:
  Bug A: leading_space_plus_one — base_bytes baked with +1 for ▁ tokens
  Bug B: byte_token_wrong_size — byte tokens get len(piece.encode()) = 6 not 1
  Bug C: missing_is_unused — boundary predicate omits sp.is_unused

Each contributes some number of bytes to the canonical denominator
inflation. This run isolates each contribution by constructing four
LUT variants (canonical + each bug applied individually + all three
together = yahya's full LUT) and measuring the ratio for each.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sentencepiece as spm

TOKENIZER = "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
VAL = "/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin"
OUT = "/workspace/agent-pgolf/audit/empirical_validation/run5_bug_decomposition.json"

SEQ_LEN = 2048
STRIDE = 64


def load_val():
    header = np.fromfile(VAL, dtype="<i4", count=256)
    assert int(header[0]) == 20240520
    n = int(header[2])
    return np.fromfile(VAL, dtype="<u2", count=n, offset=1024)


def build_lut(sp, bug_a=False, bug_b=False, bug_c=False):
    """Build a LUT with selected bug families applied.

    bug_a: leading_space_plus_one - bake +1 into base_bytes for ▁ tokens
    bug_b: byte_token_wrong_size - size byte tokens by piece UTF-8 length
    bug_c: missing_is_unused - omit sp.is_unused from boundary predicate
    """
    vocab = sp.GetPieceSize()
    base_bytes = np.zeros(vocab, dtype=np.int64)
    has_space = np.zeros(vocab, dtype=bool)
    is_boundary = np.zeros(vocab, dtype=bool)

    for i in range(vocab):
        if bug_c:
            is_boundary[i] = sp.IsControl(i) or sp.IsUnknown(i)
        else:
            is_boundary[i] = sp.IsControl(i) or sp.IsUnknown(i) or sp.IsUnused(i)

        if is_boundary[i]:
            continue

        piece = sp.IdToPiece(i)

        if sp.IsByte(i):
            if bug_b:
                base_bytes[i] = len(piece.encode("utf-8"))
            else:
                base_bytes[i] = 1
            continue

        if piece.startswith("\u2581"):
            has_space[i] = True
            stripped = piece[1:]
            if bug_a:
                base_bytes[i] = len(stripped.encode("utf-8")) + 1
            else:
                base_bytes[i] = len(stripped.encode("utf-8"))
        else:
            base_bytes[i] = len(piece.encode("utf-8"))

    return base_bytes, has_space, is_boundary


def compute_canonical_eval(val, base_bytes, has_space, is_boundary, bug_a):
    """Compute canonical_total and buggy_total under the canonical eval formula.

    For canonical / no-bug-A LUTs:
      canonical_total = base_bytes[y].sum() + (has_space[y] & ~is_boundary[x]).sum()
      buggy_total = canonical_total + eval_add (the +1 bug applied at eval)

    For bug-A LUTs (base_bytes already has +1 baked):
      canonical_total = base_bytes[y].sum()
      buggy_total = base_bytes[y].sum() + eval_add (eval applies +1 again)

    Returns (canonical_total, buggy_total, eval_add, scored_tokens).
    """
    n = val.shape[0]
    last_end = ((n - SEQ_LEN) // STRIDE) * STRIDE + SEQ_LEN
    if last_end > n:
        last_end = n
    y = val[1:last_end].astype(np.int64)
    x = val[:last_end - 1].astype(np.int64)

    eval_add = int((has_space[y] & ~is_boundary[x]).sum())
    base_total = int(base_bytes[y].sum())

    if bug_a:
        canonical_total = base_total
        buggy_total = base_total + eval_add
    else:
        canonical_total = base_total + eval_add
        buggy_total = canonical_total + eval_add

    return canonical_total, buggy_total, eval_add, int(y.shape[0])


def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER)
    val = load_val()
    print(f"[run5] val: {val.shape[0]:,} tokens, vocab: {sp.GetPieceSize()}")

    # Count vocab statistics
    n_byte_tokens = sum(1 for i in range(sp.GetPieceSize()) if sp.IsByte(i))
    n_unused_tokens = sum(1 for i in range(sp.GetPieceSize())
                          if sp.IsUnused(i) and not (sp.IsControl(i) or sp.IsUnknown(i)))
    n_leading_space = sum(1 for i in range(sp.GetPieceSize())
                          if not (sp.IsControl(i) or sp.IsUnknown(i) or sp.IsUnused(i))
                          and not sp.IsByte(i)
                          and sp.IdToPiece(i).startswith("\u2581"))
    print(f"[run5] vocab: {n_byte_tokens} byte tokens, {n_unused_tokens} unused, "
          f"{n_leading_space} leading-space pieces")
    print()

    configs = [
        ("canonical", False, False, False),
        ("only_bug_a", True, False, False),
        ("only_bug_b", False, True, False),
        ("only_bug_c", False, False, True),
        ("bugs_a_b", True, True, False),
        ("bugs_a_c", True, False, True),
        ("bugs_b_c", False, True, True),
        ("all_three", True, True, True),
    ]

    results = {}
    for name, bug_a, bug_b, bug_c in configs:
        bb, hs, ib = build_lut(sp, bug_a, bug_b, bug_c)
        c_total, b_total, ea, scored = compute_canonical_eval(val, bb, hs, ib, bug_a)
        ratio = b_total / c_total
        results[name] = {
            "bug_a_leading_space_plus_one": bug_a,
            "bug_b_byte_token_wrong_size": bug_b,
            "bug_c_missing_is_unused": bug_c,
            "canonical_bytes": c_total,
            "buggy_bytes": b_total,
            "eval_add": ea,
            "ratio": ratio,
            "scored_tokens": scored,
        }
        bug_str = "+".join([n for n, b in [("A", bug_a), ("B", bug_b), ("C", bug_c)] if b]) or "none"
        print(f"[run5] {name:15s} (bugs: {bug_str:8s}) "
              f"canonical={c_total:>11,}  buggy={b_total:>11,}  ratio={ratio:.10f}")

    # Compute per-bug isolated contributions to the inflation ratio
    canonical_ratio = results["canonical"]["ratio"]
    yahya_ratio = results["all_three"]["ratio"]

    print()
    print(f"[run5] === decomposition ===")
    print(f"[run5] canonical (no bugs):  ratio = {canonical_ratio:.6f}")
    print(f"[run5] all three bugs:       ratio = {yahya_ratio:.6f}")
    print(f"[run5] total inflation:      Δratio = {yahya_ratio - canonical_ratio:+.6f}")
    print()
    print("[run5] Per-bug isolated effect on ratio (relative to canonical):")
    for bug_name, key in [("A: leading_space +1   ", "only_bug_a"),
                          ("B: byte_token=6       ", "only_bug_b"),
                          ("C: missing_is_unused  ", "only_bug_c")]:
        delta = results[key]["ratio"] - canonical_ratio
        delta_pct = delta / (yahya_ratio - canonical_ratio) * 100 if yahya_ratio != canonical_ratio else 0
        print(f"[run5]   {bug_name}: Δratio = {delta:+.6f}  ({delta_pct:+.1f}% of total)")

    print()
    print("[run5] Per-bug byte contribution to the canonical denominator:")
    canonical_bytes = results["canonical"]["canonical_bytes"]
    for bug_name, key in [("A: leading_space +1   ", "only_bug_a"),
                          ("B: byte_token=6       ", "only_bug_b"),
                          ("C: missing_is_unused  ", "only_bug_c")]:
        delta_bytes = results[key]["canonical_bytes"] - canonical_bytes
        print(f"[run5]   {bug_name}: Δcanonical_bytes = {delta_bytes:+,}")

    output = {
        "results": results,
        "vocab_stats": {
            "n_byte_tokens": n_byte_tokens,
            "n_unused_tokens": n_unused_tokens,
            "n_leading_space": n_leading_space,
        },
        "tokenizer_path": TOKENIZER,
        "val_path": VAL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[run5] Wrote {OUT}")


if __name__ == "__main__":
    main()
