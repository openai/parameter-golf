"""Unpack bigbag's LZMA-packed train_gpt.py into readable source.

Run:  python unpack.py <packed.py> <unpacked_out.py>
"""
import base64, lzma, re, sys

packed_path, out_path = sys.argv[1], sys.argv[2]
blob = open(packed_path).read()

m = re.search(r'B\.b85decode\("([^"]+)"\)', blob)
if m is None:
    raise SystemExit(f"No base85 blob found in {packed_path}")
b85 = m.group(1)

src = lzma.decompress(
    base64.b85decode(b85),
    format=lzma.FORMAT_RAW,
    filters=[{"id": lzma.FILTER_LZMA2}],
).decode()

with open(out_path, "w", encoding="utf-8") as f:
    f.write(src)

print(f"packed   : {len(blob):>8} bytes")
print(f"unpacked : {len(src):>8} bytes  ->  {out_path}")
print(f"lines    : {src.count(chr(10)) + 1}")
