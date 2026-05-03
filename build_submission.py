"""
Build a submission-shaped artifact from readable Python source.
Compresses source with LZMA2 (raw format) and wraps in a 2-line self-extractor,
matching the format used by Kevin Clark's PR #1394 submission.

Usage:
  python build_submission.py <input.py> <output.py>

Example:
  python build_submission.py train_gpt_human_qkgain.py train_gpt_submission.py
"""
import sys
import lzma
import base64


def build(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        source_bytes = f.read()

    compressed = lzma.compress(
        source_bytes,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
    )
    encoded = base64.b85encode(compressed).decode("ascii")

    wrapper = (
        'import lzma as L,base64 as B\n'
        'exec(L.decompress(B.b85decode("'
        + encoded
        + '"),format=L.FORMAT_RAW,filters=[{"id":L.FILTER_LZMA2}]))'
    )

    with open(output_path, "w") as f:
        f.write(wrapper)

    print(f"Input size:      {len(source_bytes):,} bytes ({input_path})")
    print(f"Compressed size: {len(compressed):,} bytes")
    print(f"Output size:     {len(wrapper.encode()):,} bytes ({output_path})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1], sys.argv[2])
