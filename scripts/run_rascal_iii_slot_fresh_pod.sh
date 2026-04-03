#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/.git" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
elif [[ -d "${SCRIPT_DIR}/../.git" ]]; then
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
else
  REPO_ROOT="$(pwd)"
fi
cd "${REPO_ROOT}"

SEED="${SEED:-300}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_POD_SETUP="${RUN_POD_SETUP:-1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
SHIM_DIR="${REPO_ROOT}/.tmp/rascal_slot_shims"
TARGET_RUN_SH="${REPO_ROOT}/neural/2026-03-31_Rascal_III_SLOT/run.sh"

log() { printf '%s\n' "$*"; }
die() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

[[ -x "${PYTHON_BIN}" ]] || die "python3 not found"
[[ -f "${TARGET_RUN_SH}" ]] || die "missing run script: ${TARGET_RUN_SH}"

log "============================================"
log "  RASCAL III SLOT - FRESH POD WRAPPER"
log "  repo: ${REPO_ROOT}"
log "  seed: ${SEED}"
log "  nproc: ${NPROC_PER_NODE}"
log "============================================"

if [[ "${RUN_POD_SETUP}" == "1" ]]; then
  log "[1/4] Running scripts/pod_setup.sh ..."
  bash "${REPO_ROOT}/scripts/pod_setup.sh"
else
  log "[1/4] Skipping pod setup (RUN_POD_SETUP=${RUN_POD_SETUP})"
fi

PYTHON_BIN="$(command -v python3)"
PYTHON_DIR="$(dirname -- "${PYTHON_BIN}")"
mkdir -p "${SHIM_DIR}"
cat > "${SHIM_DIR}/torchrun" <<SHIM
#!/usr/bin/env bash
exec "${PYTHON_BIN}" -m torch.distributed.run "\$@"
SHIM
chmod +x "${SHIM_DIR}/torchrun"

export PATH="${SHIM_DIR}:${PYTHON_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

log "[2/4] Verifying live runtime ..."
"${PYTHON_BIN}" - <<'PY'
import shutil
import sys

import torch
import zstandard
from flash_attn_interface import flash_attn_func  # noqa: F401

print(f"python   : {sys.executable}")
print(f"torch    : {torch.__version__}")
print(f"cuda     : {torch.version.cuda}")
print(f"torchrun : {shutil.which('torchrun')}")
print(f"zstd     : {zstandard.__version__}")
print("fa3      : OK")

assert torch.__version__ == "2.4.1+cu124", f"wrong torch: {torch.__version__}"
assert str(torch.version.cuda).startswith("12.4"), f"wrong cuda: {torch.version.cuda}"
PY

log "[3/4] torchrun shim ..."
log "python3  -> ${PYTHON_BIN}"
log "torchrun -> $(command -v torchrun)"
head -n 1 "$(command -v torchrun)" || true

log "[4/4] Launching untouched SLOT run.sh ..."
SEED="${SEED}" NPROC_PER_NODE="${NPROC_PER_NODE}" bash "${TARGET_RUN_SH}"
