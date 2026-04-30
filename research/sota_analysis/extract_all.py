#!/usr/bin/env python3
"""Extract all merged record trainers into sota_analysis/records/<name>.py, decompressing as needed."""
import lzma, base64, re, subprocess, pathlib, sys

REPO = pathlib.Path("/Users/william/Desktop/parameter-golf")
OUT  = REPO / "sota_analysis" / "records"
OUT.mkdir(exist_ok=True)

def git_show(path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"origin/main:{path}"])

def git_ls(path: str) -> list[str]:
    out = subprocess.check_output(["git", "-C", str(REPO), "ls-tree", "--name-only", "origin/main", path]).decode()
    return [l for l in out.splitlines() if l]

# Pick the primary python entrypoint in a record directory
ENTRY_CANDIDATES = [
    "train_gpt.py",
    "train_gpt_v5.py",
    "train_gpt_cuda_ternary.py",
    "train_gpt_human.py",
]

def find_entry(record_dir: str) -> str | None:
    files = git_ls(record_dir + "/")
    for cand in ENTRY_CANDIDATES:
        full = f"{record_dir}/{cand}"
        if full in files:
            return full
    # fallback: any train*.py
    for f in files:
        if f.endswith(".py") and "train" in f.rsplit("/",1)[-1]:
            return f
    return None

def maybe_decompress(text: str, record_dir: str) -> str:
    # Chained wrapper: exec(open(__file__.replace(...)).read())
    m = re.search(r"open\(__file__\.replace\([\"']train_gpt\.py[\"'],\s*[\"']([^\"']+)[\"']\)\)\.read\(\)", text)
    if m:
        inner_name = m.group(1)
        inner = git_show(f"{record_dir}/{inner_name}").decode()
        return maybe_decompress(inner, record_dir)
    # LZMA+base85 wrapper.  Base85 includes '(' and ')' so we cannot use a
    # simple regex to find the closing paren -- walk characters while skipping
    # over string literals to find it.
    if 'b85decode' not in text:
        return text
    idx = text.index("b85decode(") + len("b85decode(")
    i = idx
    depth = 1
    in_str = None
    while i < len(text) and depth > 0:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2; continue
            if c == in_str:
                in_str = None
        else:
            if c in ('"', "'"):
                in_str = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
        i += 1
    payload = text[idx:i]
    chunks = re.findall(r"([\"'])(.*?)(?<!\\)\1", payload, re.DOTALL)
    blob = "".join(c[1] for c in chunks)
    if not blob:
        return text
    try:
        return lzma.decompress(base64.b85decode(blob),
                               format=lzma.FORMAT_RAW,
                               filters=[{"id": lzma.FILTER_LZMA2}]).decode()
    except lzma.LZMAError:
        return lzma.decompress(base64.b85decode(blob)).decode()

def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

records_root = "records/track_10min_16mb"
record_dirs = sorted(d for d in git_ls(records_root + "/") if d != records_root)

summary = []
for rd in record_dirs:
    entry = find_entry(rd)
    if entry is None:
        summary.append((rd, "NO_PYTHON_FILE", 0))
        continue
    try:
        raw = git_show(entry).decode(errors="replace")
        src = maybe_decompress(raw, rd)
    except Exception as e:
        summary.append((rd, f"ERR:{e}", 0))
        continue
    fname = slug(rd.split("/")[-1]) + ".py"
    (OUT / fname).write_text(src)
    summary.append((rd.split("/")[-1], entry.split("/")[-1], src.count("\n")+1))

for name, entry, n in summary:
    print(f"{name:65s} | {entry:32s} | {n:5d} lines")
