#!/usr/bin/env bash
set -euo pipefail

WHEEL_PATH="${1:-}"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "Usage: $0 /abs/path/to/flash_attn_3-*.whl"
  exit 1
fi

if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "FATAL: wheel not found: ${WHEEL_PATH}"
  exit 1
fi

python3 - <<'PYEOF'
import torch
tv = torch.__version__
cv = torch.version.cuda or ""
assert tv.startswith("2.4.1"), f"wrong torch: {tv}"
assert cv.startswith("12.4"), f"wrong cuda: {cv}"
print(f"torch={tv} cuda={cv}")
PYEOF

python3 -m pip install -U pip
python3 -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"
python3 -m pip install -U einops

TORCH_LIB="$(python3 - <<'PYEOF'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

python3 - <<'PYEOF'
import importlib
import torch
import flash_attn_interface
importlib.import_module("flash_attn_3._C")
print(f"flash_attn_interface={flash_attn_interface.__file__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print("FA3 wheel install check: OK")
PYEOF

echo "READY: FA3 wheel installed and verified."
