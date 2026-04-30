"""Canonical BPB byte-count audit tool for Parameter Golf.

**What it does.** A static audit of the ``build_sentencepiece_luts`` byte-count
bug in Parameter Golf PRs descended from the #1698 lineage. The tool
classifies each ``train_gpt.py`` as CORRECT / BUGGY / OBFUSCATED / UNKNOWN,
and for non-obfuscated scripts it computes the canonical and buggy byte
totals on the exact scored-token subset the eval loop would use. The
inflation ratio is ``buggy / canonical``; for a BUGGY script the inferred
canonical BPB is ``reported_bpb * inflation_ratio``.

**What it does NOT do.** The tool only inspects the byte-count LUT. It does
not verify that ``eval_val_sliding`` itself is canonical (the eval loop is
assumed faithful; differences there are out of scope). It does not verify
that a reported BPB was produced by the submitted ``train_gpt.py`` against
an unmodified val shard — the arithmetic correction assumes the numerator
(cross-entropy loss in nats) was correctly measured by the submitter. It
does not validate the trained model artifact, hyperparameters, or any
other aspect of submission integrity beyond the LUT.

**Algorithm.**
1. Regex-classify the LUT: look for ``len(piece.encode("utf-8")) + 1``
   (BUGGY), the bare ``len(piece.encode("utf-8"))`` assignment (CORRECT),
   or a ``*.decompress(*.b85decode(...))`` wrapper (OBFUSCATED).
2. For non-obfuscated scripts, build the canonical LUT from the SP model
   and the scored-token subset from the val shard.
3. Collapse the per-window byte sum into two array reductions over
   ``val_tokens[1:N]``; the buggy total is ``canonical + sum(has_leading_space[y])``.

**Example usage.**
::

    python scripts/canonical_rescore.py \\
        --train-script <pr-train_gpt.py> \\
        --tokenizer    data/tokenizers/fineweb_8192_bpe.model \\
        --val-data     'data/datasets/fineweb10B_sp8192/fineweb_val_*.bin' \\
        --reported-bpb 1.02840 \\
        --pr-number    1758

See ``scripts/README_canonical_rescore.md`` for a full CLI reference and
``audit/methodology.md`` for the math derivation (in particular §4 on why
the inflation ratio depends on the scoring strategy).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Static LUT classification
# ---------------------------------------------------------------------------

# Obfuscated submissions wrap the entire module body in either
# ``exec(lzma.decompress(base64.b85decode(...)))`` or assign the decoded blob
# to a local and execute it via ``runpy``/``exec`` later. Both share a single
# expression chaining ``decompress(...b85decode(...))`` — match that, not bare
# imports (PR #1727 imports lzma for an artifact compressor without being
# obfuscated).
_OBFUSCATED_RE = re.compile(
    r"[A-Za-z_][\w.]*\.decompress\s*\(\s*[A-Za-z_][\w.]*\.b85decode\s*\(",
    re.DOTALL,
)

# --- Property 1: leading-space base_bytes assignment ("+1 or not") ---------
# The canonical upstream form is
#   base_bytes_np[token_id] = len(piece.encode("utf-8"))
# after stripping the leading ▁. The #1698 buggy form bakes a +1 into the LUT:
#   base_bytes_np[token_id] = len(piece.encode("utf-8")) + 1
# yahya010's PR #1734 train_gdn_7k.py uses a slice-based variant:
#   base_bytes[i] = len(piece[1:].encode("utf-8")) + 1
# so we accept any ``len(<expr>.encode("utf-8"))`` where <expr> is a simple
# identifier or subscript (no nested parens). ``[^()\n]*`` enforces this.
_LEN_ENCODE_UTF8 = r"len\(\s*[^()\n]*\.encode\s*\(\s*['\"]utf-8['\"]\s*\)\s*\)"
_LEADING_PLUS1_RE = re.compile(
    r"base_bytes[\w]*\s*\[[^\]]+\]\s*=\s*" + _LEN_ENCODE_UTF8 + r"\s*\+\s*1"
)
_LEADING_NOPLUS_RE = re.compile(
    r"base_bytes[\w]*\s*\[[^\]]+\]\s*=\s*" + _LEN_ENCODE_UTF8 + r"(?!\s*\+\s*1)"
)

# --- Property 2: sp.is_byte branch assigns literal 1 -----------------------
# The ``if sp.is_byte(<id>):`` branch can be followed by either an inline
# assignment (``base_bytes[i] = 1``) on the next indented line or a block
# with several statements before ``continue``. We look at the next 1-6
# indented lines for a ``base_bytes[...] = <rhs>`` assignment.
_IS_BYTE_BRANCH_RE = re.compile(
    r"if\s+(?:sp|tokenizer|tok|_sp|spm)?\.?is_byte\s*\(\s*[^)]+\)\s*:\s*\n"
    r"(?P<body>(?:[ \t]+[^\n]*\n){1,6})"
)
_BYTE_TOKEN_ASSIGN_RE = re.compile(
    r"base_bytes[\w]*\s*\[[^\]]+\]\s*=\s*(?P<rhs>[^\n#]+)"
)

# --- Property 3: boundary predicate includes is_unused --------------------
# The canonical boundary line looks like
#     if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
# We detect sites by the presence of ``is_control(`` and check the nearby
# window for ``is_unknown`` and ``is_unused``.


def _detect_leading_space(src: str) -> str:
    """P1 detector: ``base_bytes = len(piece.encode("utf-8"))`` vs ``... + 1``."""
    if _LEADING_PLUS1_RE.search(src):
        return "DEVIATES"
    if _LEADING_NOPLUS_RE.search(src):
        return "MATCHES_CANONICAL"
    return "INDETERMINATE"


def _detect_byte_token(src: str) -> str:
    """P2 detector: ``if sp.is_byte(...): base_bytes = 1``.

    Returns ``DEVIATES`` only when a ``sp.is_byte(...)`` branch is located and
    the assignment inside it is something other than literal ``1`` (e.g.
    ``len(piece.encode("utf-8"))``). If no ``sp.is_byte`` branch is found at
    all, returns ``INDETERMINATE`` — the function may handle byte tokens in a
    different idiom we do not parse, or not at all.
    """
    m = _IS_BYTE_BRANCH_RE.search(src)
    if not m:
        return "INDETERMINATE"
    body = m.group("body")
    assign = _BYTE_TOKEN_ASSIGN_RE.search(body)
    if not assign:
        return "INDETERMINATE"
    rhs = assign.group("rhs").strip().rstrip(";")
    if rhs == "1":
        return "MATCHES_CANONICAL"
    return "DEVIATES"


_IS_UNKNOWN_CALL_RE = re.compile(r"is_unknown\s*\(")
_IS_UNUSED_CALL_RE = re.compile(r"is_unused\s*\(")


def _detect_boundary_predicate(src: str) -> str:
    """P3 detector: boundary predicate includes ``sp.is_unused``.

    Scans every occurrence of ``is_control(`` in the source, grabs a window
    around it, and checks whether ``is_unknown(`` and ``is_unused(`` calls
    appear (requiring the opening paren so comment text mentioning
    "is_unused" does not confuse the detector).

    * If any such window contains both ``is_unknown(`` and ``is_unused(``:
      ``MATCHES_CANONICAL``.
    * Else if any contains ``is_unknown(`` but no ``is_unused(``:
      ``DEVIATES`` (canonical boundary missing ``is_unused``).
    * Else (no ``is_control(`` at all, or no ``is_unknown(`` nearby):
      ``INDETERMINATE``.
    """
    any_boundary_like = False
    for m in re.finditer(r"is_control\s*\(", src):
        start = max(0, m.start() - 120)
        end = min(len(src), m.end() + 300)
        window = src[start:end]
        if not _IS_UNKNOWN_CALL_RE.search(window):
            continue
        any_boundary_like = True
        if _IS_UNUSED_CALL_RE.search(window):
            return "MATCHES_CANONICAL"
    if any_boundary_like:
        return "DEVIATES"
    return "INDETERMINATE"


# Human-readable descriptions for each deviation name. Keyed by the strings
# that appear in ``lut_bug_detections``.
BUG_DESCRIPTIONS = {
    "leading_space_plus_one":
        "Bakes +1 into LUT for leading-space tokens, causing eval_val_sliding "
        "to double-count the leading-space byte (#1698 lineage bug).",
    "byte_token_wrong_size":
        "sp.is_byte branch sizes byte tokens by len(piece.encode('utf-8')) "
        "(= 6 for '<0xXX>') instead of the canonical literal 1.",
    "missing_is_unused":
        "Boundary predicate omits sp.is_unused; unused tokens are scored as "
        "if they contributed bytes instead of being treated as zero-byte "
        "boundaries.",
}


def classify_lut_detailed(src: str) -> tuple[str, list[str]]:
    """Classify a ``train_gpt.py`` and return the list of deviating properties.

    Args:
        src: The full contents of the ``train_gpt.py`` file as a string.

    Returns:
        A tuple ``(status, deviations)``. ``status`` is one of
        ``CORRECT`` / ``BUGGY`` / ``OBFUSCATED`` / ``UNKNOWN``. ``deviations``
        is a list of property-name strings drawn from
        ``{"leading_space_plus_one", "byte_token_wrong_size",
        "missing_is_unused"}``.

    Classification rules:
        * Any property DEVIATES ⟹ ``BUGGY``. ``deviations`` lists which.
        * All three properties MATCH canonical ⟹ ``CORRECT``.
        * No deviations AND the obfuscation regex matches ⟹ ``OBFUSCATED``.
        * Otherwise ⟹ ``UNKNOWN``.

    Gotchas:
        LUT pattern detection takes priority over obfuscation detection: a
        script can import ``lzma`` or call ``base64.b85decode`` legitimately
        (e.g. PR #1727 ships a JS minifier as a compressed blob) without
        being obfuscated. We only flag ``OBFUSCATED`` when no deviations are
        found AND the three canonical properties do not all match AND the
        source contains a chained ``*.decompress(*.b85decode(...))``
        expression. Both inline-``exec`` and assign-then-``runpy`` wrapper
        styles are handled.
    """
    p1 = _detect_leading_space(src)
    p2 = _detect_byte_token(src)
    p3 = _detect_boundary_predicate(src)

    deviations: list[str] = []
    if p1 == "DEVIATES":
        deviations.append("leading_space_plus_one")
    if p2 == "DEVIATES":
        deviations.append("byte_token_wrong_size")
    if p3 == "DEVIATES":
        deviations.append("missing_is_unused")

    if deviations:
        return "BUGGY", deviations
    if p1 == "MATCHES_CANONICAL" and p2 == "MATCHES_CANONICAL" and p3 == "MATCHES_CANONICAL":
        return "CORRECT", []
    if _OBFUSCATED_RE.search(src):
        return "OBFUSCATED", []
    return "UNKNOWN", []


def classify_lut(src: str) -> str:
    """Classify a ``train_gpt.py`` source string. Returns status only.

    See ``classify_lut_detailed`` for the richer (status, deviations) return.
    """
    return classify_lut_detailed(src)[0]


# ---------------------------------------------------------------------------
# Tokenizer LUT construction (canonical, no +1)
# ---------------------------------------------------------------------------


def build_canonical_luts(tokenizer_path: Path, vocab_size: Optional[int] = None):
    """Build the canonical SentencePiece byte LUTs used by ``eval_val_sliding``.

    Args:
        tokenizer_path: Path to the SentencePiece ``.model`` file.
        vocab_size: Optional override. If larger than the SP vocab, the arrays
            are padded with zeros to that size (matches upstream behaviour
            when the model's vocab is smaller than the padded embedding).

    Returns:
        A tuple ``(base_bytes, has_leading_space, is_boundary)`` of numpy
        arrays, shape ``[table_size]``. ``base_bytes`` stores the canonical
        UTF-8 byte length per token (with leading ``▁`` stripped and no +1);
        ``has_leading_space`` marks pieces that begin with ``▁``;
        ``is_boundary`` marks control/unknown/unused tokens.

    Gotchas:
        This is the canonical "no +1 in LUT" version. The +1 for leading
        spaces is added at eval time, gated by ``~is_boundary[x_prev]``. A
        LUT that bakes the +1 in (the #1698 bug) double-counts when combined
        with the standard eval loop.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size or sp_vocab)

    base_bytes = np.zeros(table_size, dtype=np.int32)
    has_leading_space = np.zeros(table_size, dtype=bool)
    is_boundary = np.ones(table_size, dtype=bool)

    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):  # SentencePiece ▁
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))

    return base_bytes, has_leading_space, is_boundary


# ---------------------------------------------------------------------------
# Validation token loading
# ---------------------------------------------------------------------------


def load_val_tokens(pattern: str) -> np.ndarray:
    """Load fineweb val shards into a 1-D numpy array of uint16 token ids.

    Args:
        pattern: Glob, explicit path, or directory. For a directory the tool
            expands to ``fineweb_val_*.bin``.

    Returns:
        A flat numpy array of uint16 token ids, shards concatenated in sorted
        order.

    Gotchas:
        Mirrors ``load_data_shard`` in the upstream ``train_gpt.py``. The
        shard format is a 256-int32 header (magic ``20240520``, version
        ``1``, token count, 253 zero-padded) followed by ``n`` little-endian
        uint16 tokens. Shards that don't match the magic/version raise.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        p = Path(pattern)
        if p.exists():
            paths = [str(p)]
        elif p.is_dir():
            paths = sorted(str(x) for x in p.glob("fineweb_val_*.bin"))
    if not paths:
        raise FileNotFoundError(f"No val files matched: {pattern}")
    chunks = []
    for path in paths:
        header = np.fromfile(path, dtype="<i4", count=256)
        if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
            raise ValueError(f"Unexpected shard header for {path}")
        n = int(header[2])
        toks = np.fromfile(path, dtype="<u2", count=n, offset=256 * 4)
        if toks.size != n:
            raise ValueError(f"Short read for {path}: expected {n} got {toks.size}")
        chunks.append(toks)
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


# ---------------------------------------------------------------------------
# Sliding-window byte computation
# ---------------------------------------------------------------------------


SCORING_MODES = (
    "sliding-window-boundary-masked",
    "all-tokens-boundary-masked",
    "all-tokens-no-mask",
)


@dataclass
class ByteCountResult:
    canonical_byte_count: int
    buggy_byte_count: int
    leading_space_token_count: int
    scored_token_count: int
    num_windows: int
    scoring_mode: str = "sliding-window-boundary-masked"


def compute_byte_counts(
    val_tokens: np.ndarray,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary: np.ndarray,
    seq_len: int,
    stride: int,
    scoring_mode: str = "sliding-window-boundary-masked",
) -> ByteCountResult:
    """Compute canonical and buggy byte totals under the chosen scoring mode.

    Three modes are supported:

    * ``sliding-window-boundary-masked`` (default): scored y-tokens = the exact
      subset the upstream ``eval_val_sliding`` in PR #1727 actually scores
      (``seq_len=2048, stride=64`` windows, last window trimmed to end of val).
      Leading-space bytes are gated by ``~is_boundary[x_prev]`` — the same gate
      the eval loop applies. This is what PR #1727's eval pipeline reports.
    * ``all-tokens-boundary-masked``: scored y-tokens = every position in the
      flat slice ``val_tokens[1:N]``. Same boundary-mask gate. On val data
      where the sliding windows already tile the full stream (the SP8192 case),
      this is numerically identical to sliding-window-boundary-masked.
    * ``all-tokens-no-mask``: scored y-tokens = flat ``val_tokens[1:N]`` slice,
      with boundary_mask = 1 everywhere (every leading-space byte is counted).
      This corresponds to the "decode the whole stream and count UTF-8 bytes"
      naive ground-truth that yahya010 used in the PR #1734 closure note.

    The buggy byte total always equals canonical + ``sum(has_leading_space[y])``
    regardless of the mask — the LUT adds +1 per leading-space token, and the
    eval still adds the gated +1 on top, so the per-token delta is exactly one.
    The inflation *ratio* varies because the canonical denominator varies.
    """
    if val_tokens.ndim != 1:
        raise ValueError("val_tokens must be 1-D")
    if scoring_mode not in SCORING_MODES:
        raise ValueError(f"unknown scoring_mode {scoring_mode!r}; valid: {SCORING_MODES}")
    total_tokens = int(val_tokens.shape[0]) - 1
    context_size = seq_len - stride
    if context_size < 0:
        raise ValueError(f"seq_len ({seq_len}) must be >= stride ({stride})")

    if scoring_mode.startswith("sliding-window"):
        # Replicate upstream window selection for the window count + tile end.
        window_starts = [
            ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens
        ]
        num_windows = len(window_starts)
        if num_windows == 0:
            return ByteCountResult(0, 0, 0, 0, 0, scoring_mode=scoring_mode)
        last_ws = window_starts[-1]
        last_end = min(last_ws + seq_len, total_tokens)
        expected_scored = last_end
    else:
        # "all-tokens-*" variants score every position in val_tokens[1:N].
        num_windows = 0
        expected_scored = total_tokens

    y = val_tokens[1 : expected_scored + 1].astype(np.int64, copy=False)
    x = val_tokens[0 : expected_scored].astype(np.int64, copy=False)

    bb = base_bytes[y].astype(np.int64)
    ls = has_leading_space[y]
    if scoring_mode.endswith("no-mask"):
        mask = np.ones_like(ls)
    else:
        pb = is_boundary[x]
        mask = ~pb
    canonical_total = int(bb.sum()) + int((ls & mask).sum())
    leading_space_total = int(ls.sum())
    buggy_total = canonical_total + leading_space_total

    return ByteCountResult(
        canonical_byte_count=canonical_total,
        buggy_byte_count=buggy_total,
        leading_space_token_count=leading_space_total,
        scored_token_count=int(expected_scored),
        num_windows=num_windows,
        scoring_mode=scoring_mode,
    )


# ---------------------------------------------------------------------------
# Top-level rescore entrypoint
# ---------------------------------------------------------------------------


def rescore(
    train_script: Path,
    tokenizer: Path,
    val_data: str,
    seq_len: int = 2048,
    stride: int = 64,
    reported_bpb: Optional[float] = None,
    pr_number: Optional[int] = None,
    threshold: float = 1.0738,
    max_val_tokens: Optional[int] = None,
    skip_byte_count: bool = False,
    scoring_mode: str = "sliding-window-boundary-masked",
) -> dict:
    """End-to-end LUT classification + byte-count rescore.

    Args:
        train_script: Path to the candidate ``train_gpt.py``.
        tokenizer: Path to the matching SentencePiece ``.model``.
        val_data: Glob / path / directory for fineweb val ``.bin`` shards.
        seq_len, stride: Upstream eval-loop parameters (default 2048 / 64).
        reported_bpb: Submitter-reported ``val_bpb``. If given and the script
            is BUGGY, ``inferred_canonical_bpb = reported_bpb * ratio``.
        pr_number: Optional int to embed in the output JSON.
        threshold: Upper bound for ``passes_merged_sota_threshold`` (default
            1.0738 — one record-class margin under the current merged SOTA).
        max_val_tokens: Truncate the val stream to this many tokens (for
            fast smoke tests; must NOT be set for an audit run).
        skip_byte_count: Classify the LUT only; do not load val data.
        scoring_mode: One of ``SCORING_MODES`` — see ``compute_byte_counts``.

    Returns:
        A dict with the LUT classification, byte totals, inflation ratio,
        inferred canonical BPB, and threshold verdict. Full schema is in
        ``scripts/README_canonical_rescore.md``.
    """
    src = train_script.read_text(errors="replace")
    lut_status, lut_bug_detections = classify_lut_detailed(src)

    counts: Optional[ByteCountResult] = None
    inflation_ratio: Optional[float] = None
    notes: list[str] = []

    if lut_status == "OBFUSCATED":
        notes.append("Code is lzma/b85-obfuscated; LUT cannot be verified statically.")
    elif lut_status == "UNKNOWN":
        notes.append(
            "build_sentencepiece_luts pattern not recognized; manual review required."
        )

    if not skip_byte_count and lut_status != "OBFUSCATED":
        base_bytes, has_leading_space, is_boundary = build_canonical_luts(tokenizer)
        val_tokens = load_val_tokens(val_data)
        if max_val_tokens is not None and val_tokens.shape[0] > max_val_tokens:
            val_tokens = val_tokens[:max_val_tokens]
            notes.append(f"Truncated val tokens to {max_val_tokens} for fast inspection.")
        counts = compute_byte_counts(
            val_tokens, base_bytes, has_leading_space, is_boundary, seq_len, stride,
            scoring_mode=scoring_mode,
        )
        if counts.canonical_byte_count > 0:
            inflation_ratio = counts.buggy_byte_count / counts.canonical_byte_count

    # Apply the inflation only when the LUT has the leading_space_plus_one
    # deviation — that is the specific arithmetic the ratio math computes.
    # A BUGGY PR with only byte_token_wrong_size or missing_is_unused
    # deviations requires a separate LUT rebuild for an arithmetic correction.
    applied_ratio: Optional[float]
    inflation_ratio_includes: list[str]
    if lut_status == "CORRECT":
        applied_ratio = 1.0
        inflation_ratio_includes = []
    elif lut_status == "BUGGY":
        if "leading_space_plus_one" in lut_bug_detections:
            applied_ratio = inflation_ratio
            inflation_ratio_includes = ["leading_space_plus_one"]
            if len(lut_bug_detections) > 1:
                other = [b for b in lut_bug_detections if b != "leading_space_plus_one"]
                notes.append(
                    "inflation_ratio accounts only for leading_space_plus_one; "
                    "additional deviations present (" + ", ".join(other) + ") "
                    "would increase the canonical correction further — the reported "
                    "inferred_canonical_bpb is therefore conservative (an underestimate "
                    "of the true canonical BPB)."
                )
        else:
            applied_ratio = None
            inflation_ratio_includes = []
            notes.append(
                "BUGGY but no leading_space_plus_one deviation; the +1 "
                "inflation arithmetic does not apply. A PR-specific LUT "
                "rebuild is required for an arithmetic BPB correction."
            )
    else:
        applied_ratio = None
        inflation_ratio_includes = []

    inferred_canonical_bpb: Optional[float] = None
    if reported_bpb is not None and applied_ratio is not None:
        inferred_canonical_bpb = reported_bpb * applied_ratio

    passes_threshold: Optional[bool] = None
    if inferred_canonical_bpb is not None:
        passes_threshold = inferred_canonical_bpb <= threshold

    detected_bugs_description = "; ".join(
        BUG_DESCRIPTIONS[name] for name in lut_bug_detections if name in BUG_DESCRIPTIONS
    )

    result = {
        "pr_number": pr_number,
        "script_path": str(train_script),
        "lut_status": lut_status,
        "lut_bug_detections": lut_bug_detections,
        "detected_bugs_description": detected_bugs_description,
        "inflation_ratio_includes": inflation_ratio_includes,
        "reported_bpb": reported_bpb,
        "inflation_ratio": applied_ratio,
        "computed_inflation_ratio": inflation_ratio,
        "inferred_canonical_bpb": inferred_canonical_bpb,
        "passes_merged_sota_threshold": passes_threshold,
        "merged_sota_threshold": threshold,
        "seq_len": seq_len,
        "stride": stride,
        "scoring_mode": scoring_mode,
    }
    if counts is not None:
        result["canonical_byte_count"] = counts.canonical_byte_count
        result["buggy_byte_count"] = counts.buggy_byte_count
        result["leading_space_token_count"] = counts.leading_space_token_count
        result["scored_token_count"] = counts.scored_token_count
        result["num_windows"] = counts.num_windows
    if notes:
        result["notes"] = "; ".join(notes)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train-script", type=Path, required=True,
                   help="Path to the candidate train_gpt.py to inspect.")
    p.add_argument("--tokenizer", type=Path, required=True,
                   help="Path to the matching SentencePiece .model file.")
    p.add_argument("--val-data", type=str, required=True,
                   help="Glob or path for fineweb val .bin shards (e.g. "
                        "'data/datasets/fineweb10B_sp8192/fineweb_val_*.bin').")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Sliding-window length, matching eval_val_sliding (default 2048).")
    p.add_argument("--stride", type=int, default=64,
                   help="Sliding-window stride, matching eval_val_sliding (default 64).")
    p.add_argument("--reported-bpb", type=float, default=None,
                   help="Submitter-reported val_bpb. When set with a BUGGY "
                        "script, the tool emits inferred_canonical_bpb = "
                        "reported_bpb * inflation_ratio.")
    p.add_argument("--pr-number", type=int, default=None,
                   help="Optional PR number to embed in the output JSON.")
    p.add_argument("--threshold", type=float, default=1.0738,
                   help="Upper bound for passes_merged_sota_threshold "
                        "(default 1.0738 — one record-class margin below SOTA).")
    p.add_argument("--max-val-tokens", type=int, default=None,
                   help="Truncate val data (for fast smoke tests; do not use for audit)")
    p.add_argument("--skip-byte-count", action="store_true",
                   help="Only run static LUT classification; skip the byte computation")
    p.add_argument("--scoring-mode", type=str, default="sliding-window-boundary-masked",
                   choices=list(SCORING_MODES),
                   help=(
                       "Which y-token subset + boundary-mask policy to use for the "
                       "byte totals. 'sliding-window-boundary-masked' (default) "
                       "mirrors PR #1727's eval_val_sliding exactly and yields the "
                       "ratio the eval pipeline would report. 'all-tokens-no-mask' "
                       "mirrors yahya010's 'decode the full stream' ground-truth "
                       "used in the PR #1734 closure. See audit/methodology.md §4."
                   ))
    p.add_argument("--output", type=Path, default=None,
                   help="Write JSON to this path (in addition to stdout)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = rescore(
        train_script=args.train_script,
        tokenizer=args.tokenizer,
        val_data=args.val_data,
        seq_len=args.seq_len,
        stride=args.stride,
        reported_bpb=args.reported_bpb,
        pr_number=args.pr_number,
        threshold=args.threshold,
        max_val_tokens=args.max_val_tokens,
        skip_byte_count=args.skip_byte_count,
        scoring_mode=args.scoring_mode,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
