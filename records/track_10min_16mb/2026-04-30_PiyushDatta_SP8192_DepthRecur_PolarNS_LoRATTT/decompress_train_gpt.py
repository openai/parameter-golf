#!/usr/bin/env python3
"""Decompress train_gpt.py (LZMA-compressed) into train_gpt_readable.py for human reference."""
import lzma, base64, sys
from pathlib import Path

script_dir = Path(__file__).parent
src = script_dir / "train_gpt.py"
dst = script_dir / "train_gpt_readable.py"

content = src.read_text()
start = content.index('b85decode("') + len('b85decode("')
end = content.index('")', start)
decompressed = lzma.decompress(
    base64.b85decode(content[start:end]),
    format=lzma.FORMAT_RAW,
    filters=[{"id": lzma.FILTER_LZMA2}],
)
dst.write_bytes(decompressed)
print(f"Decompressed {len(decompressed)} bytes -> {dst}")
