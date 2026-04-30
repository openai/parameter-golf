#!/usr/bin/env bash
# Phase 0b: unpack bigbag's LZMA-compressed train_gpt.py into readable source.
# Expects unpack.py in the current directory (uploaded alongside this script).
set -euo pipefail

REC=records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
cd /workspace/parameter-golf

echo "--- unpack baseline ---"
python /workspace/unpack.py "$REC/train_gpt.py" /workspace/work/train_gpt_baseline.py

echo "--- readable? first 30 lines ---"
head -30 /workspace/work/train_gpt_baseline.py

echo "--- stats ---"
wc -l /workspace/work/train_gpt_baseline.py
md5sum /workspace/work/train_gpt_baseline.py

echo "--- does it at least import cleanly? ---"
# guard against accidental top-level training code
python - <<'PY'
import ast, pathlib
src = pathlib.Path("/workspace/work/train_gpt_baseline.py").read_text()
tree = ast.parse(src)
defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
print(f"top-level defs: {len(defs)}")
print("sample:", defs[:20])
PY

echo "=== PHASE 0b DONE ==="
