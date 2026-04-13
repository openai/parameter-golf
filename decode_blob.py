#!/usr/bin/env python3
"""Decode lzma+ascii85 blob from blob.py to train_gpt_from_blob.py (no exec)."""
from __future__ import annotations

import base64 as B
import lzma as L
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def extract_b85(text: str) -> str:
    m = re.search(r"B\.b85decode\(\s*\"([^\"]+)\"\s*\)", text)
    if m:
        return m.group(1)
    i = text.find('b85decode("')
    if i < 0:
        raise ValueError("No b85decode(\"...\") blob found in blob.py")
    i += len('b85decode("')
    j = text.find('"),format=L.FORMAT_RAW', i)
    if j < 0:
        j = text.find("'),format=L.FORMAT_RAW", i)
    if j < 0:
        raise ValueError("Could not find end of b85 string")
    return text[i:j]


def main() -> None:
    blob_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "blob.py"
    out_path = ROOT / "train_gpt_from_blob.py"
    text = blob_path.read_text(encoding="utf-8")
    if len(text.strip()) < 100:
        print(
            f"{blob_path} is empty or too short on disk. "
            "Save blob.py in the editor (Cmd+S), or pass the path to a saved copy, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    s = extract_b85(text)
    raw = L.decompress(
        B.b85decode(s.encode("ascii")),
        format=L.FORMAT_RAW,
        filters=[{"id": L.FILTER_LZMA2}],
    )
    out_path.write_bytes(raw)
    print(f"Wrote {out_path} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
