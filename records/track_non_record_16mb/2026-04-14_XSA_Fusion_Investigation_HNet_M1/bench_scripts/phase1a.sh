#!/usr/bin/env bash
# Phase 1a: check FA3 availability + show the hot classes (Block, CausalSelfAttention, MLP, GPT).
set -euo pipefail
cd /workspace/parameter-golf

echo "--- is FA3 importable? ---"
python - <<'PY' || true
try:
    import flash_attn_interface as fa3
    print("FA3 OK", getattr(fa3, "__version__", "?"), "module:", fa3.__file__)
    from flash_attn_interface import flash_attn_func
    print("flash_attn_func sig:", flash_attn_func.__doc__[:200] if flash_attn_func.__doc__ else "(no doc)")
except Exception as e:
    print("FA3 MISSING:", type(e).__name__, e)
PY

echo
echo "--- installed flash-attn-ish packages ---"
pip list 2>/dev/null | grep -i -E 'flash|attn' || echo "(none)"

echo
echo "--- extract the hot classes for inspection ---"
python - <<'PY'
import ast, pathlib, textwrap
src = pathlib.Path("/workspace/work/train_gpt_baseline.py").read_text()
tree = ast.parse(src)
wanted = {"CausalSelfAttention", "MLP", "Block", "GPT"}
lines = src.splitlines()
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in wanted:
        start, end = node.lineno - 1, node.end_lineno
        body = "\n".join(lines[start:end])
        print(f"\n=== class {node.name} @ lines {node.lineno}-{node.end_lineno} ({end-start} lines) ===")
        print(body)
PY

echo
echo "--- what does the forward pass look like? show GPT.forward ---"
python - <<'PY'
import ast, pathlib
src = pathlib.Path("/workspace/work/train_gpt_baseline.py").read_text()
tree = ast.parse(src)
lines = src.splitlines()
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "GPT":
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name == "forward":
                start, end = m.lineno - 1, m.end_lineno
                print("\n".join(lines[start:end]))
PY

echo "=== PHASE 1a DONE ==="
