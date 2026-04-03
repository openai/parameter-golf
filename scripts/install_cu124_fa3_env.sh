#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-fa3wheel}"
VENV_DIR="${VENV_DIR:-/workspace/venv_cu124}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
WHEEL_PATH="${WHEEL_PATH:-${REPO_ROOT}/wheels/fa3_cu124_vast/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl}"

log() { printf '%s\n' "$*"; }
die() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

[[ -f "${WHEEL_PATH}" ]] || die "missing FA3 wheel: ${WHEEL_PATH}"

if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    log "[1/4] creating conda env ${CONDA_ENV}"
    conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip
  else
    log "[1/4] reusing conda env ${CONDA_ENV}"
  fi
  conda activate "${CONDA_ENV}"
else
  log "[1/4] reusing venv ${VENV_DIR}"
  if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

log "[2/4] installing exact cu124 stack"
python -m pip install -U pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" \
  torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124
python -m pip install \
  sentencepiece zstandard huggingface-hub datasets tiktoken attr einops ninja packaging sympy==1.12
python -m pip install --no-deps --force-reinstall "${WHEEL_PATH}"

log "[3/4] writing activation helper"
cat > "${REPO_ROOT}/scripts/activate_flywheel_env.sh" <<ACTEOF
#!/usr/bin/env bash
set -euo pipefail
if [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi
TORCH_LIB=\$(python - <<'PYEOF'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PYEOF
)
export LD_LIBRARY_PATH="\${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
export COMPILE_ENABLED=1
export COMPILE_FULLGRAPH=1
export TORCHDYNAMO_SUPPRESS_ERRORS=0
ACTEOF
chmod +x "${REPO_ROOT}/scripts/activate_flywheel_env.sh"

log "[4/4] verifying stack"
VERIFY_DATA=0 bash "${REPO_ROOT}/scripts/verify_cu124_fa3_env.sh"

log "READY"
log "Next: SEED=300 NPROC_PER_NODE=8 bash scripts/run_rascal_slot_locked.sh"
