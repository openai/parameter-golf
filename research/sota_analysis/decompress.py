#!/usr/bin/env python3
"""Decompress train_gpt.py files wrapped as exec(lzma.decompress(base85.decode(...)))."""
import lzma, base64, re, sys, pathlib

def decompress(src_path: str) -> str:
    text = pathlib.Path(src_path).read_text()
    # Not wrapped? Return as-is.
    if 'lzma' not in text[:200] or 'b85decode' not in text[:200]:
        return text
    # Match either "..." or '...' payload
    m = re.search(r"b85decode\(([\"'])(.+?)\1\)", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"Could not find base85 blob in {src_path}")
    blob = m.group(2)
    # Two wrapper flavors seen in the wild:
    # 1) raw LZMA2 filter:  format=L.FORMAT_RAW, filters=[{"id":L.FILTER_LZMA2}]
    # 2) plain lzma.decompress (xz/alone container)
    try:
        return lzma.decompress(
            base64.b85decode(blob),
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2}],
        ).decode()
    except lzma.LZMAError:
        return lzma.decompress(base64.b85decode(blob)).decode()

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = decompress(src)
    pathlib.Path(dst).write_text(out)
    print(f"{src} -> {dst}: {out.count(chr(10))+1} lines, {len(out)} bytes")
