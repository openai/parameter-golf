#!/usr/bin/env python3
"""Extract LZMA-compressed trainer source from record train_gpt.py wrappers.

Many top records wrap the real script as:
  exec(L.decompress(B.b85decode("...")), format=L.FORMAT_RAW, filters=[...])

This tool decompresses to stdout or -o without executing the trainer.

Usage:
  python3 scripts/decompress_record_train_gpt.py \\
    records/.../train_gpt.py -o /tmp/train_gpt_plain.py
"""

from __future__ import annotations

import argparse
import lzma as L
import base64 as B
import sys
from pathlib import Path


def decompress_file(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = 'b85decode("'
    start = text.find(marker)
    if start < 0:
        raise ValueError("Expected b85decode(\"...\") wrapper (not a plain train_gpt.py)")
    start += len(marker)
    end = text.find('"),format=L.FORMAT_RAW', start)
    if end < 0:
        raise ValueError("Expected ),format=L.FORMAT_RAW after b85 payload")
    blob = B.b85decode(text[start:end])
    return L.decompress(blob, format=L.FORMAT_RAW, filters=[{"id": L.FILTER_LZMA2}])


def main() -> None:
    ap = argparse.ArgumentParser(description="Decompress record train_gpt.py LZMA wrapper")
    ap.add_argument("train_gpt", type=Path, help="Path to record train_gpt.py")
    ap.add_argument("-o", "--output", type=Path, help="Write decompressed Python here (default: stdout)")
    args = ap.parse_args()
    out = decompress_file(args.train_gpt)
    if args.output:
        args.output.write_bytes(out)
        print(f"Wrote {len(out)} bytes to {args.output}", file=sys.stderr)
    else:
        sys.stdout.buffer.write(out)


if __name__ == "__main__":
    main()
