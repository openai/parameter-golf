#!/usr/bin/env bash
set -euo pipefail

# Fresh pod bootstrap for parameter-golf-lab.
# - Installs/uses miniconda env
# - Installs CUDA PyTorch + repo deps
# - Clones or syncs repo branch
# - Runs preflight checks
#
# Usage (on new pod):
#   bash scripts/bootstrap_fresh_pod.sh
#
# Common overrides:
#   BRANCH=TEST_LAB WORKSPACE=/workspace REPO_URL=https://github.com/newjordan/parameter-golf.git bash scripts/bootstrap_fresh_pod.sh
#   INSTALL_DATASET=1 TRAIN_SHARDS=1 bash scripts/bootstrap_fresh_pod.sh

REPO_URL="${REPO_URL:-https://github.com/newjordan/parameter-golf.git}"
BRANCH="${BRANCH:-TEST_LAB}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/parameter-golf-lab}"
MINICONDA_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-pglab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_DATASET="${INSTALL_DATASET:-0}"
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
FORCE_SYNC="${FORCE_SYNC:-0}"

# PyTorch install mode:
# - conda: install via conda channels
# - pip: install via pip CUDA wheels
TORCH_INSTALL_MODE="${TORCH_INSTALL_MODE:-pip}"
PIP_TORCH_INDEX_URL="${PIP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
PIP_TORCH_PACKAGES="${PIP_TORCH_PACKAGES:-torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1}"
REQUIRED_TORCH_VERSION="${REQUIRED_TORCH_VERSION:-2.4.1+cu124}"
REQUIRED_CUDA_PREFIX="${REQUIRED_CUDA_PREFIX:-12.4}"
REQUIRE_FA3="${REQUIRE_FA3:-1}"
ALLOW_FA3_WHEEL_INSTALL="${ALLOW_FA3_WHEEL_INSTALL:-0}"

mkdir -p "${WORKSPACE}"

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

ensure_cmd() {
    command -v "$1" >/dev/null 2>&1 || { echo "FATAL: missing command '$1'"; exit 1; }
}

log "Bootstrap start"
log "repo=${REPO_URL} branch=${BRANCH} repo_dir=${REPO_DIR}"
log "conda_env=${CONDA_ENV} python=${PYTHON_VERSION}"

ensure_cmd git
ensure_cmd curl
ensure_cmd bash

if [ ! -x "${MINICONDA_DIR}/bin/conda" ]; then
    log "Installing Miniconda at ${MINICONDA_DIR}"
    INSTALLER="/tmp/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"
fi

# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    log "Creating conda env ${CONDA_ENV}"
    conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${CONDA_ENV}"

log "Python in env: $(python -V)"
python -m pip install --upgrade pip setuptools wheel

if [ "${TORCH_INSTALL_MODE}" = "conda" ]; then
    log "Installing CUDA PyTorch via conda channels"
    conda install -y -n "${CONDA_ENV}" pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
else
    log "Installing CUDA PyTorch via pip index ${PIP_TORCH_INDEX_URL}"
    python -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true
    python -m pip install --no-cache-dir --force-reinstall --index-url "${PIP_TORCH_INDEX_URL}" ${PIP_TORCH_PACKAGES}
fi

if [ -d "${REPO_DIR}/.git" ]; then
    log "Repo exists at ${REPO_DIR}"
    cd "${REPO_DIR}"
    if [ "${FORCE_SYNC}" = "1" ]; then
        log "Force syncing to origin/${BRANCH}"
        git fetch origin "${BRANCH}"
        git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
        git clean -fd
    else
        log "Fast syncing branch ${BRANCH}"
        git fetch origin "${BRANCH}"
        git checkout "${BRANCH}" || git checkout -b "${BRANCH}" "origin/${BRANCH}"
        git pull --ff-only origin "${BRANCH}"
    fi
else
    log "Cloning repo"
    git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
fi

log "Installing repo deps"
python -m pip install -r requirements.txt
python -m pip install zstandard

FA3_LOCAL_PYTHONPATH="${REPO_DIR}/flash-attention/hopper:${PYTHONPATH:-}"
FA3_SYSTEM_PYTHONPATH="${PYTHONPATH:-}"
FA3_SELECTED_PYTHONPATH=""

check_fa3_path() {
PYTHONPATH="${1:-}" python - <<'PY'
from flash_attn_interface import flash_attn_func  # noqa: F401
PY
}

# Required FA3 install (no fallback)
if check_fa3_path "${FA3_LOCAL_PYTHONPATH}" >/dev/null 2>&1; then
    FA3_SELECTED_PYTHONPATH="${FA3_LOCAL_PYTHONPATH}"
    log "Using FA3 provider: local hopper path"
elif check_fa3_path "${FA3_SYSTEM_PYTHONPATH}" >/dev/null 2>&1; then
    FA3_SELECTED_PYTHONPATH="${FA3_SYSTEM_PYTHONPATH}"
    log "Using FA3 provider: system/site-packages"
else
    if [ "${ALLOW_FA3_WHEEL_INSTALL}" = "1" ]; then
        log "flash_attn_interface missing; attempting FA3 wheel (required)"
        python -m pip install --no-cache-dir \
          "https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
          || { echo "FATAL: FA3 unavailable; refusing non-SOTA fallback stack"; exit 1; }
    else
        echo "FATAL: FA3 unavailable from custom/system paths and wheel install is disabled (ALLOW_FA3_WHEEL_INSTALL=0)."
        exit 1
    fi
    if check_fa3_path "${FA3_LOCAL_PYTHONPATH}" >/dev/null 2>&1; then
        FA3_SELECTED_PYTHONPATH="${FA3_LOCAL_PYTHONPATH}"
        log "Using FA3 provider: local hopper path"
    elif check_fa3_path "${FA3_SYSTEM_PYTHONPATH}" >/dev/null 2>&1; then
        FA3_SELECTED_PYTHONPATH="${FA3_SYSTEM_PYTHONPATH}"
        log "Using FA3 provider: system/site-packages"
    else
        echo "FATAL: FA3 import failed (missing or ABI mismatch, e.g. undefined symbol)."
        exit 1
    fi
fi

mkdir -p "${REPO_DIR}/logs"

# Helpful activation helper for future shells.
cat > "${WORKSPACE}/activate_pglab.sh" <<ACTEOF
#!/usr/bin/env bash
set -euo pipefail
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${REPO_DIR}"
export PATH="${MINICONDA_DIR}/bin:\${PATH}"
export PYTHONPATH="${FA3_SELECTED_PYTHONPATH}"
ACTEOF
chmod +x "${WORKSPACE}/activate_pglab.sh"

log "Preflight checks"
python - <<'PY'
import os
import torch
print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("gpu0:", torch.cuda.get_device_name(0))

required_torch = os.environ.get("REQUIRED_TORCH_VERSION", "2.4.1+cu124")
required_cuda = os.environ.get("REQUIRED_CUDA_PREFIX", "12.4")
if torch.__version__ != required_torch:
    raise SystemExit(f"FATAL: wrong torch {torch.__version__} (expected {required_torch})")
if not (torch.version.cuda or "NONE").startswith(required_cuda):
    raise SystemExit(f"FATAL: wrong CUDA {torch.version.cuda} (expected {required_cuda}x)")
PY

if [ "${REQUIRE_FA3}" = "1" ]; then
    PYTHONPATH="${FA3_SELECTED_PYTHONPATH}" python - <<'PY'
from flash_attn_interface import flash_attn_func  # noqa: F401
print("flash_attn_interface: OK")
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
fi

log "torchrun path: $(command -v torchrun)"
head -n 1 "$(command -v torchrun)" || true

if [ "${INSTALL_DATASET}" = "1" ]; then
    log "Downloading cached challenge FineWeb (train_shards=${TRAIN_SHARDS})"
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
fi

cat <<MSG

============================================================
BOOTSTRAP COMPLETE
============================================================
Repo: ${REPO_DIR}
Branch: ${BRANCH}
Conda env: ${CONDA_ENV}
Activate helper: ${WORKSPACE}/activate_pglab.sh

Next:
  source ${WORKSPACE}/activate_pglab.sh
  bash experiments/Rascal_AB_1p109_to_1p102/run_ab_h100_2000.sh

Optional data pull:
  INSTALL_DATASET=1 TRAIN_SHARDS=1 bash scripts/bootstrap_fresh_pod.sh
============================================================
MSG
