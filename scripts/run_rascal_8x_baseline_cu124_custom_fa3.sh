#!/usr/bin/env bash
set -euo pipefail

# One-shot runner: rebuild cu124 venv, reuse system custom FA3 module, run locked Rascal baseline.
# Usage:
#   bash scripts/run_rascal_8x_baseline_cu124_custom_fa3.sh
# Optional env:
#   REBUILD_VENV=0 VENV_DIR=.venv-cu124 BASE_PYTHON=python3 SEED=444 NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${BASE_PYTHON:=python3}"
: "${VENV_DIR:=.venv-cu124}"
: "${REBUILD_VENV:=1}"
: "${NPROC_PER_NODE:=8}"
: "${SEED:=444}"
: "${MAX_WALLCLOCK_SECONDS:=600}"

echo "[preflight] locating custom FA3 module from base python: ${BASE_PYTHON}"
FA3_DIR="$("${BASE_PYTHON}" - <<'PY'
import inspect
import os
import sys
try:
    import flash_attn_interface
except Exception as e:
    print(f"ERROR:{e}")
    sys.exit(2)
print(os.path.dirname(inspect.getfile(flash_attn_interface)))
PY
)"
if [[ "${FA3_DIR}" == ERROR:* ]]; then
  echo "FATAL: base python cannot import flash_attn_interface (${FA3_DIR#ERROR:})"
  exit 1
fi
echo "[preflight] FA3_DIR=${FA3_DIR}"

if [ "${REBUILD_VENV}" = "1" ]; then
  echo "[setup] rebuilding ${VENV_DIR}"
  deactivate 2>/dev/null || true
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] creating ${VENV_DIR}"
  "${BASE_PYTHON}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
python -m pip install numpy zstandard sentencepiece

echo "[verify] torch"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
if ! python - <<'PY'
import torch
import sys
ver = str(torch.version.cuda)
if not ver.startswith("12.4"):
    print(f"FATAL: expected cu124 torch, got {torch.__version__} cuda={ver}")
    sys.exit(1)
print("cu124_ok")
PY
then
  exit 1
fi

echo "[verify] FA3 via PYTHONPATH bridge"
if ! PYTHONPATH="${FA3_DIR}:${PYTHONPATH:-}" python -c "from flash_attn_interface import flash_attn_func; print('FA3_OK_CUSTOM')"; then
  echo "FATAL: venv could not import flash_attn_interface from FA3_DIR=${FA3_DIR}"
  exit 1
fi

echo "[run] locked Rascal baseline (SKIP_GPTQ=1)"
PYTHONPATH="${FA3_DIR}:${PYTHONPATH:-}" \
PYTHON_BIN="${REPO_ROOT}/${VENV_DIR}/bin/python" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
bash scripts/run_rascal_8x_baseline_locked.sh

