import lzma
import sys

path = "records/track_10min_16mb/2026-03-25_LongContext4096_Int6_QAT/final_model.int6.ptz"
code_size = 90453

with open(path, "rb") as f:
    blob_l6 = f.read()

raw = lzma.decompress(blob_l6)
blob_l9 = lzma.compress(raw, preset=9)

print(f"lzma=6: {len(blob_l6):,} bytes  total={len(blob_l6)+code_size:,}")
print(f"lzma=9: {len(blob_l9):,} bytes  total={len(blob_l9)+code_size:,}")
print(f"Saved:  {len(blob_l6)-len(blob_l9):,} bytes")
print(f"Fits:   {len(blob_l9)+code_size < 16_000_000}")

if len(blob_l9) + code_size < 16_000_000:
    out = path.replace(".int6.ptz", ".int6.l9.ptz")
    with open(out, "wb") as f:
        f.write(blob_l9)
    print(f"Saved to {out}")
else:
    print("Still over 16MB with lzma=9")
