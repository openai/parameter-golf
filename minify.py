#!/usr/bin/env python3
"""Minify train_gpt.py for submission using python-minifier.

Usage:
    python minify.py [input] [output]
    python minify.py  # defaults: submission train_gpt.py → train_gpt.min.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import python_minifier

DEFAULT_INPUT = "records/track_10min_16mb/2026-03-29_FullStack_TTT_Ngram_KNN_TurboQuant/train_gpt.py"


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_INPUT)
    output_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".min.py")
    )

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    original = input_path.read_text(encoding="utf-8")
    minified = python_minifier.minify(
        original,
        remove_annotations=True,
        remove_pass=True,
        remove_literal_statements=True,  # strips docstrings
        combine_imports=True,
        hoist_literals=False,  # don't hoist — can break torch.compile
        rename_locals=False,  # don't rename — breaks getattr/env var lookups
        rename_globals=False,
        remove_object_base=True,
        convert_posargs_to_args=False,
        preserve_shebang=False,
    )

    # Verify
    try:
        ast.parse(minified)
    except SyntaxError as e:
        print(f"ERROR: Minification produced invalid Python: {e}", file=sys.stderr)
        sys.exit(1)

    output_path.write_text(minified, encoding="utf-8")

    orig_bytes = len(original.encode("utf-8"))
    mini_bytes = len(minified.encode("utf-8"))
    saved = orig_bytes - mini_bytes
    pct = (saved / orig_bytes) * 100

    print(
        f"Input:   {input_path} ({orig_bytes:,} bytes, {len(original.splitlines())} lines)"
    )
    print(
        f"Output:  {output_path} ({mini_bytes:,} bytes, {len(minified.splitlines())} lines)"
    )
    print(f"Saved:   {saved:,} bytes ({pct:.1f}%)")


if __name__ == "__main__":
    main()
