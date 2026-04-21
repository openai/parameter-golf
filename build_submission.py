"""
Build LZMA-compressed submission from train_gpt.py.
Outputs a 2-line train_gpt.py that decompresses and execs the real code.
"""
import lzma
import base64
import sys
from pathlib import Path


def build_submission(input_path="train_gpt.py", output_path="records/submission/train_gpt.py"):
    source = Path(input_path).read_text(encoding="utf-8")
    source_bytes = source.encode("utf-8")
    print(f"Original: {len(source_bytes)} bytes, {source.count(chr(10))} lines")

    # Compress with LZMA2 (raw format for smallest output)
    compressed = lzma.compress(
        source_bytes,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2}],
    )
    print(f"LZMA2 compressed: {len(compressed)} bytes ({len(compressed)/len(source_bytes)*100:.1f}%)")

    # Base85 encode (more compact than base64)
    encoded = base64.b85encode(compressed).decode("ascii")
    print(f"Base85 encoded: {len(encoded)} bytes")

    # Build the 2-line wrapper
    wrapper = (
        'import lzma as L,base64 as B\n'
        f'exec(L.decompress(B.b85decode("{encoded}"),'
        f'format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))'
    )

    wrapper_bytes = len(wrapper.encode("utf-8"))
    print(f"Final wrapper: {wrapper_bytes} bytes (this is the code_bytes for submission)")
    print(f"Budget remaining for model: {16_000_000 - wrapper_bytes} bytes")

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(wrapper, encoding="utf-8")
    print(f"Written to {out}")

    # Verify roundtrip
    exec_globals = {}
    recovered = lzma.decompress(
        base64.b85decode(encoded),
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2}],
    )
    assert recovered == source_bytes, "Roundtrip verification FAILED!"
    print("Roundtrip verification OK")

    return wrapper_bytes


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "train_gpt.py"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "records/submission/train_gpt.py"
    build_submission(input_path, output_path)
