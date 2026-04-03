#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HOPPER_DIR="${REPO_ROOT}/flash-attention/hopper"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/wheels/fa3_cu124}"
MAX_JOBS="${MAX_JOBS:-4}"
TMPDIR="${TMPDIR:-/workspace/tmp}"

if [[ ! -d "${HOPPER_DIR}" ]]; then
  echo "FATAL: missing ${HOPPER_DIR}"
  exit 1
fi

mkdir -p "${OUT_DIR}" "${TMPDIR}"

python3 - <<'PYEOF'
import torch
tv = torch.__version__
cv = torch.version.cuda or ""
assert tv.startswith("2.4.1"), f"wrong torch: {tv}"
assert cv.startswith("12.4"), f"wrong cuda: {cv}"
print(f"torch={tv} cuda={cv}")
PYEOF

cd "${HOPPER_DIR}"

# Historical known-good FA3 trim profile (no aggressive pruning).
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export MAX_JOBS
export TMPDIR

python3 -m pip install -U pip setuptools wheel ninja packaging

rm -f "${OUT_DIR}"/flash_attn_3-*.whl "${OUT_DIR}"/flash_attn_3-*.whl.sha256 "${OUT_DIR}"/build_manifest.txt
python3 -m pip wheel . --no-build-isolation --no-deps -w "${OUT_DIR}"

WHEEL_PATH="$(ls -t "${OUT_DIR}"/flash_attn_3-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "FATAL: wheel build did not produce flash_attn_3-*.whl"
  exit 1
fi

python3 -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"

TORCH_LIB="$(python3 - <<'PYEOF'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYEOF
)"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

python3 - <<'PYEOF'
import importlib
importlib.import_module("flash_attn_3._C")
from flash_attn_interface import flash_attn_func  # noqa: F401
print("FA3 wheel runtime check: OK")
PYEOF

sha256sum "${WHEEL_PATH}" > "${WHEEL_PATH}.sha256"

python3 - <<PYEOF > "${OUT_DIR}/build_manifest.txt"
import os
import platform
import torch
wheel = os.path.basename("${WHEEL_PATH}")
print(f"wheel={wheel}")
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"max_jobs=${MAX_JOBS}")
print("flags=FLASH_ATTENTION_DISABLE_HDIM96,FLASH_ATTENTION_DISABLE_FP8,FLASH_ATTENTION_DISABLE_VARLEN,FLASH_ATTENTION_DISABLE_SM80")
PYEOF

echo "WHEEL_PATH=${WHEEL_PATH}"
echo "SHA256_PATH=${WHEEL_PATH}.sha256"
echo "MANIFEST_PATH=${OUT_DIR}/build_manifest.txt"
echo "READY: FA3 cu124 wheel built and verified."
