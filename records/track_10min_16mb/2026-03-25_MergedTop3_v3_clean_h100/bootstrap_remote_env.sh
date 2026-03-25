#!/usr/bin/env bash
set -euo pipefail

export REMOTE_VENV_DIR="${REMOTE_VENV_DIR:-/workspace/.venvs/parameter-golf-20260325}"
export REMOTE_PIP_CACHE="${REMOTE_PIP_CACHE:-/workspace/.cache/pip}"
export TMPDIR="${TMPDIR:-/workspace/.cache/tmp}"

mkdir -p "${REMOTE_PIP_CACHE}"
mkdir -p "${TMPDIR}"

if [[ ! -x "${REMOTE_VENV_DIR}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${REMOTE_VENV_DIR}"
fi

source "${REMOTE_VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null
python -m pip install --cache-dir "${REMOTE_PIP_CACHE}" -r requirements-remote.txt >/dev/null

if [[ "${DISABLE_FLASH_ATTN:-0}" != "1" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
import importlib
for name in ("flash_attn_interface", "flash_attn"):
    try:
        importlib.import_module(name)
        raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
  then
    python -m pip install --no-cache-dir --no-build-isolation flash-attn >/dev/null
  fi
fi

python - <<'PY'
import importlib

required = ["torch", "sentencepiece", "datasets", "zstandard"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit(f"bootstrap failed: missing imports after install: {', '.join(missing)}")
print("bootstrap ok")
PY
