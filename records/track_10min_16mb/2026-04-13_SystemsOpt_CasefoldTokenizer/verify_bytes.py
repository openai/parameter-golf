#!/usr/bin/env python3
"""
verify_bytes.py — Verify that the casefold v2 tokenizer counts bytes correctly.

No GPU required. Two modes:

  Full corpus (results included in submission):
    uv run verify_bytes.py --docs data/docs_selected.jsonl --max-docs 0

  Spot check (bundled 200-doc sample):
    python verify_bytes.py --docs verify_docs.jsonl

Checks:
  1. Zero uppercase tokens in the casefold vocabulary
  2. LUT byte count == ground-truth UTF-8 bytes on every document

Exit code 0 = all checks pass. Exit code 1 = a check failed.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Output tee — print to stdout and optionally save to file
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, file=None):
        self._file = file
        import io
        if hasattr(sys.stdout, 'buffer'):
            self._stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        else:
            self._stdout = sys.stdout

    def write(self, msg):
        self._stdout.write(msg)
        if self._file:
            self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        if self._file:
            self._file.flush()


# ─────────────────────────────────────────────────────────────────────────────
# LUT builder — mirrors build_sentencepiece_luts() in train_gpt_human.py
# ─────────────────────────────────────────────────────────────────────────────

def build_lut(sp):
    """Build byte-counting lookup tables from a SentencePiece model.

    Returns (base_bytes, has_leading_space, is_boundary) as plain Python lists.
    Same algorithm as train_gpt_human.py, without torch dependencies.
    """
    n = sp.vocab_size()
    base_bytes = [0] * n
    has_leading_space = [False] * n
    is_boundary = [True] * n

    for tid in range(n):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith('\u2581'):
            has_leading_space[tid] = True
            piece = piece[1:]
        piece = piece.replace('\u2581', ' ')
        base_bytes[tid] = len(piece.encode('utf-8'))

    return base_bytes, has_leading_space, is_boundary


def count_bytes_lut(token_ids, base_bytes, has_leading_space, is_boundary):
    """Count bytes from a token sequence using LUT.

    Note: in eval_val(), byte counting starts at token 1 because token 0
    is context-only (not predicted). Here we count ALL tokens since we're
    verifying against the full text's byte count.
    """
    if not token_ids:
        return 0
    total = base_bytes[token_ids[0]]
    for i in range(1, len(token_ids)):
        tid = token_ids[i]
        prev = token_ids[i - 1]
        total += base_bytes[tid]
        if has_leading_space[tid] and not is_boundary[prev]:
            total += 1
    return total


def normalize_casefold(text):
    """NFKC + lowercase — same normalization as retokenize.py --normalize casefold."""
    return unicodedata.normalize("NFKC", text).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming doc iterator
# ─────────────────────────────────────────────────────────────────────────────

def iter_docs(path, max_docs=0):
    """Yield (index, doc_dict) from a JSONL file, one doc at a time."""
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield count, json.loads(line)
            count += 1
            if max_docs and count >= max_docs:
                break


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_no_uppercase(sp, out):
    """CHECK 1: Verify zero uppercase characters in learned pieces."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("CHECK 1: Zero uppercase tokens in casefold vocabulary\n")
    out.write("=" * 70 + "\n")

    uppercase_tokens = []
    learned = 0
    for tid in range(sp.vocab_size()):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid) or sp.is_byte(tid):
            continue
        learned += 1
        piece = sp.id_to_piece(tid)
        cleaned = piece.lstrip('\u2581')
        if cleaned != cleaned.lower():
            uppercase_tokens.append((tid, piece))

    if uppercase_tokens:
        out.write(f"  FAIL: Found {len(uppercase_tokens)} tokens with uppercase:\n")
        for tid, piece in uppercase_tokens[:20]:
            out.write(f"    [{tid}] {repr(piece)}\n")
        return False

    out.write(f"  PASS: 0 uppercase tokens found in {learned} learned pieces\n")
    return True


def check_lut_bytes(docs_path, max_docs, sp, out):
    """CHECK 2: Verify LUT byte count == ground-truth bytes on every document.

    Ground truth = byte count of the text after SentencePiece's internal
    normalization (nmt_nfkc: NFKC + whitespace collapsing). We use
    sp.normalize() to get the exact text SP tokenizes, then count bytes.
    This is the text the model actually sees and predicts on.
    """
    out.write(f"\n{'=' * 70}\n")
    out.write("CHECK 2: LUT byte count vs ground-truth bytes (per document)\n")
    out.write("=" * 70 + "\n")

    base_bytes, has_leading_space, is_boundary = build_lut(sp)

    n_docs = 0
    ground_truth_total = 0
    lut_total = 0
    tokens_total = 0
    mismatched = []

    for i, doc in iter_docs(docs_path, max_docs):
        text = doc['text']
        norm_text = normalize_casefold(text)
        # Use SP's own normalizer to get the exact text it tokenizes.
        # sp.normalize() returns text with ▁ (U+2581) for spaces;
        # replace with ASCII space before counting bytes.
        sp_normalized = sp.normalize(norm_text).replace('\u2581', ' ')
        true_bytes = len(sp_normalized.encode('utf-8'))
        ground_truth_total += true_bytes

        token_ids = sp.encode(norm_text)
        tokens_total += len(token_ids)

        lut = count_bytes_lut(token_ids, base_bytes, has_leading_space, is_boundary)
        lut_total += lut

        if lut != true_bytes:
            mismatched.append((i, true_bytes, lut, lut - true_bytes,
                               text[:80].replace('\n', ' ')))

        n_docs = i + 1
        if n_docs % 10_000 == 0:
            out.write(f"  ... processed {n_docs:,} docs\n")
            out.flush()

    # Report
    out.write(f"\n  Documents:         {n_docs:,}\n")
    out.write(f"  Tokens:            {tokens_total:,}\n")
    out.write(f"  Ground-truth bytes:{ground_truth_total:,}\n")
    out.write(f"  LUT bytes:         {lut_total:,}\n")
    out.write(f"  Tok/byte:          {tokens_total / ground_truth_total:.6f}\n")
    out.write(f"  Mismatched docs:   {len(mismatched):,} / {n_docs:,}\n")

    stats = {'n_docs': n_docs, 'tokens': tokens_total,
             'ground_truth': ground_truth_total, 'lut': lut_total,
             'mismatched': len(mismatched)}

    if len(mismatched) == 0:
        out.write(f"\n  PASS: LUT exactly matches ground truth on every document\n")
        return True, stats

    out.write(f"\n  Per-doc mismatches (first 20):\n")
    for idx, gt, lut_val, diff, snippet in mismatched[:20]:
        out.write(f"    doc[{idx}]: ground_truth={gt:,} lut={lut_val:,} "
                  f"diff={diff:+,}  \"{snippet}...\"\n")
    pct = (lut_total - ground_truth_total) / ground_truth_total * 100
    out.write(f"\n  FAIL: LUT differs on {len(mismatched)} docs ({pct:+.6f}% total)\n")
    return False, stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify byte counting for casefold tokenizer submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--docs', type=Path, default=None,
                        help='Path to docs JSONL (verify_docs.jsonl or full corpus)')
    parser.add_argument('--casefold-model', type=Path, default=None,
                        help='Path to casefold tokenizer .model file')
    parser.add_argument('--max-docs', type=int, default=0,
                        help='Max docs to process (0 = all)')
    parser.add_argument('--save-results', type=Path, default=None,
                        help='Save output to file')
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Find docs
    docs_path = args.docs
    if docs_path is None:
        for candidate in [script_dir / 'verify_docs.jsonl', Path('verify_docs.jsonl')]:
            if candidate.exists():
                docs_path = candidate
                break
    if docs_path is None or not docs_path.exists():
        print("ERROR: docs file not found. Specify with --docs")
        sys.exit(1)

    # Find tokenizer
    cf_model = args.casefold_model
    if cf_model is None:
        for candidate in [
            script_dir / 'fineweb_8192_bpe_casefold_refined_v2.model',
            Path('data/tokenizers/fineweb_8192_bpe_casefold_refined_v2.model'),
        ]:
            if candidate.exists():
                cf_model = candidate
                break
    if cf_model is None or not cf_model.exists():
        print("ERROR: Casefold tokenizer not found. Specify with --casefold-model")
        sys.exit(1)

    # Output
    results_file = None
    if args.save_results:
        results_file = open(args.save_results, 'w', encoding='utf-8')
    out = Tee(results_file)

    sp = spm.SentencePieceProcessor(model_file=str(cf_model))

    # Header
    out.write("Casefold Tokenizer Byte Verification\n")
    out.write("=" * 70 + "\n")
    max_docs_str = 'all' if args.max_docs == 0 else f'{args.max_docs:,}'
    out.write(f"  Docs:        {docs_path}\n")
    out.write(f"  Max docs:    {max_docs_str}\n")
    out.write(f"  Tokenizer:   {cf_model}\n")
    out.write(f"  Vocab size:  {sp.vocab_size()}\n")

    all_pass = True

    # Check 1
    if not check_no_uppercase(sp, out):
        all_pass = False

    # Check 2
    ok, stats = check_lut_bytes(docs_path, args.max_docs, sp, out)
    if not ok:
        all_pass = False

    # Summary
    out.write(f"\n{'=' * 70}\n")
    out.write("SUMMARY\n")
    out.write("=" * 70 + "\n")
    out.write(f"  Documents verified:  {stats['n_docs']:,}\n")
    out.write(f"  Ground-truth bytes:  {stats['ground_truth']:,}\n")
    out.write(f"  LUT bytes:           {stats['lut']:,}\n")
    out.write(f"  LUT == ground truth: {'yes' if stats['mismatched'] == 0 else 'NO'}\n")
    out.write(f"  Tokens:              {stats['tokens']:,}\n")
    out.write(f"  Tok/byte:            {stats['tokens'] / stats['ground_truth']:.6f}\n")
    out.write("\n")

    if all_pass:
        out.write("RESULT: ALL CHECKS PASSED\n")
        out.write("The casefold v2 tokenizer counts bytes correctly.\n")
    else:
        out.write("RESULT: VERIFICATION FAILED (see details above)\n")

    if results_file:
        out.write(f"\nResults saved to {args.save_results}\n")
        results_file.close()

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
