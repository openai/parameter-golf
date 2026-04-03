#!/usr/bin/env bash
set -euo pipefail

# One-shot runner: rebuild cu124 venv, reuse system custom FA3 module, run locked Rascal baseline.
# Usage:
#   bash scripts/run_rascal_8x_baseline_cu124_custom_fa3.sh
# Optional env:
#   REBUILD_VENV=0 VENV_DIR=.venv-cu124 BASE_PYTHON=python3 SEED=444 NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${BASE_PYTHON:=}"
: "${VENV_DIR:=.venv-cu124}"
: "${REBUILD_VENV:=1}"
: "${NPROC_PER_NODE:=8}"
: "${SEED:=444}"
: "${MAX_WALLCLOCK_SECONDS:=600}"

echo "[preflight] locating custom FA3 module from a non-venv python"
if [ -n "${BASE_PYTHON}" ]; then
  candidates=("${BASE_PYTHON}")
else
  mapfile -t _which_py < <(which -a python3 2>/dev/null | awk '!seen[$0]++')
  candidates=("${_which_py[@]}")
  candidates+=(
    "/usr/bin/python3"
    "/opt/conda/bin/python3"
    "/opt/conda/bin/python"
    "/usr/local/bin/python3"
    "/usr/local/bin/python"
    "/root/miniconda3/bin/python3"
    "/root/miniconda3/bin/python"
    "/workspace/miniconda3/bin/python3"
    "/workspace/miniconda3/bin/python"
  )
  for p in /opt/conda/envs/*/bin/python3 /opt/conda/envs/*/bin/python /root/.conda/envs/*/bin/python3 /root/.conda/envs/*/bin/python; do
    candidates+=("${p}")
  done
fi

FA3_BASE_PYTHON=""
FA3_DIR=""
for p in "${candidates[@]}"; do
  [ -x "${p}" ] || continue
  if out="$("${p}" - <<'PY' 2>/dev/null
import inspect
import os
import flash_attn_interface
print(os.path.dirname(inspect.getfile(flash_attn_interface)))
PY
)"; then
    FA3_BASE_PYTHON="${p}"
    FA3_DIR="${out}"
    break
  fi
done

if [ -z "${FA3_BASE_PYTHON}" ]; then
  # Fallback: discover module file directly in common roots (py/so).
  while IFS= read -r mpath; do
    [ -n "${mpath}" ] || continue
    FA3_DIR="$(dirname "${mpath}")"
    break
  done < <(find /workspace /opt/conda /usr/local /root -type f \( -name flash_attn_interface.py -o -name "flash_attn_interface*.so" \) 2>/dev/null | head -n 1)
  if [ -z "${FA3_DIR}" ]; then
    echo "[preflight] custom FA3 not found; will try wheel fallback inside ${VENV_DIR} with --no-deps"
  fi
fi
echo "[preflight] FA3 base python: ${FA3_BASE_PYTHON}"
echo "[preflight] FA3_DIR=${FA3_DIR}"

if [ "${REBUILD_VENV}" = "1" ]; then
  echo "[setup] rebuilding ${VENV_DIR}"
  deactivate 2>/dev/null || true
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] creating ${VENV_DIR}"
  "${FA3_BASE_PYTHON:-${BASE_PYTHON:-python3}}" -m venv "${VENV_DIR}"
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
if [ -n "${FA3_DIR}" ] && PYTHONPATH="${FA3_DIR}:${PYTHONPATH:-}" python -c "from flash_attn_interface import flash_attn_func; print('FA3_OK_CUSTOM')"; then
  FA3_MODE="custom"
else
  echo "[setup] attempting FA3 wheel fallback (--no-deps)"
  if python -m pip install --no-deps --no-cache-dir \
      "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
      || python -m pip install --no-deps --no-cache-dir \
      "https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"; then
    python -c "from flash_attn_interface import flash_attn_func; print('FA3_OK_WHEEL')"
    FA3_MODE="wheel"
    FA3_DIR=""
  else
    echo "FATAL: FA3 unavailable (custom import missing and wheel fallback failed)."
    exit 1
  fi
fi

echo "[run] locked Rascal baseline (SKIP_GPTQ=1)"
PYTHONPATH="${FA3_DIR:+${FA3_DIR}:}${PYTHONPATH:-}" \
PYTHON_BIN="${REPO_ROOT}/${VENV_DIR}/bin/python" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
bash scripts/run_rascal_8x_baseline_locked.sh
