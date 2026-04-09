import lzma, pathlib
_src = lzma.decompress(pathlib.Path(__file__).with_suffix('.py.lzma').read_bytes())
exec(compile(_src, __file__, 'exec'))
