from pathlib import Path
import lzma as L
import base64 as B
import py_compile

src_path = Path("train_gpt_patched.py")
if not src_path.exists():
    raise FileNotFoundError("train_gpt_patched.py not found")

src = src_path.read_bytes()

filters = [{"id": L.FILTER_LZMA2, "preset": 9}]
compressed = L.compress(src, format=L.FORMAT_RAW, filters=filters)
encoded = B.b85encode(compressed).decode("ascii")

wrapper = (
    "import lzma as L,base64 as B\n"
    "exec(L.decompress(B.b85decode('"
    + encoded +
    "'),format=L.FORMAT_RAW,filters=[{'id':L.FILTER_LZMA2}]).decode('utf-8'))\n"
)

out = Path("train_gpt_py311_wrapper.py")
out.write_text(wrapper)

print("source bytes:", len(src))
print("compressed bytes:", len(compressed))
print("wrapper bytes:", len(wrapper.encode("utf-8")))

py_compile.compile(str(out), doraise=True)
print("wrapper syntax ok:", out.resolve())

# Verify payload syntax too.
payload = L.decompress(
    B.b85decode(encoded),
    format=L.FORMAT_RAW,
    filters=[{"id": L.FILTER_LZMA2}],
)
Path("_wrapper_payload_check.py").write_bytes(payload)
py_compile.compile("_wrapper_payload_check.py", doraise=True)
print("payload syntax ok")
