import lzma, json, os, sys
with open(os.path.join(os.path.dirname(__file__), "code.lzma"), "rb") as f:
    bundle = json.loads(lzma.decompress(f.read()))
for name, code in bundle.items():
    with open(os.path.join(os.path.dirname(__file__), name), "w") as f:
        f.write(code)
exec(compile(bundle["train_gpt_mega.py"], "train_gpt_mega.py", "exec"))
