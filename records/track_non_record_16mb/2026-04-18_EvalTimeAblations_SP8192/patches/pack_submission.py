"""Pack train_gpt_stacked_v2_fixed.py into a 2-line LZMA self-extracting submission.

Output: train_gpt.py (the actual submission file).
When executed via `torchrun ... train_gpt.py`, it decompresses and execs the merged Path A v3 code.
"""
import lzma as L, base64 as B
import python_minifier

src_file = 'train_gpt_stacked_v2_fixed.py'
out_file = 'train_gpt.py'

raw = open(src_file).read()
print(f"raw: {len(raw):,} bytes")

# Minify. Use conservative settings so we don't break semantic behavior.
minified = python_minifier.minify(
    raw,
    remove_annotations=True,
    remove_pass=True,
    remove_literal_statements=True,
    combine_imports=True,
    hoist_literals=True,
    rename_locals=True,
    rename_globals=False,  # safer — don't rename module-level names
    remove_asserts=False,  # keep asserts for safety
    remove_debug=False,
    remove_object_base=True,
    convert_posargs_to_args=True,
    preserve_shebang=False,
)
print(f"minified: {len(minified):,} bytes")

compressed = L.compress(
    minified.encode('utf-8'),
    format=L.FORMAT_RAW,
    filters=[{"id": L.FILTER_LZMA2, "preset": 9 | L.PRESET_EXTREME}],
)
print(f"lzma: {len(compressed):,} bytes")

b85 = B.b85encode(compressed).decode('ascii')

wrapper = (
    f'import lzma as L,base64 as B\n'
    f'exec(L.decompress(B.b85decode("{b85}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))\n'
)
print(f"wrapped: {len(wrapper):,} bytes")

open(out_file, 'w').write(wrapper)
import py_compile
py_compile.compile(out_file, doraise=True)
print(f"saved + syntax OK: {out_file}")
