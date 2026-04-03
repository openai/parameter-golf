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
# - conda: install via conda channels (default, most reliable on fresh pods)
# - pip: install via pip CUDA wheels
TORCH_INSTALL_MODE="${TORCH_INSTALL_MODE:-conda}"
PIP_TORCH_INDEX_URL="${PIP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

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
    python -m pip install --upgrade torch torchvision torchaudio --index-url "${PIP_TORCH_INDEX_URL}"
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

if [ -x "${REPO_DIR}/scripts/pod_setup.sh" ]; then
    log "Running strict FA3 setup (scripts/pod_setup.sh)"
    bash "${REPO_DIR}/scripts/pod_setup.sh"
else
    echo "FATAL: missing ${REPO_DIR}/scripts/pod_setup.sh (required for strict cu124+FA3 setup)"
    exit 1
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
ACTEOF
chmod +x "${WORKSPACE}/activate_pglab.sh"

log "Preflight checks"
python - <<'PY'
import os
import importlib
import torch
from flash_attn_interface import flash_attn_func  # noqa: F401
print("python:", os.sys.executable)
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("gpu0:", torch.cuda.get_device_name(0))
importlib.import_module("flash_attn_3._C")
print("fa3_runtime: OK")
assert torch.__version__ == "2.4.1+cu124", f"wrong torch: {torch.__version__}"
assert str(torch.version.cuda).startswith("12.4"), f"wrong cuda: {torch.version.cuda}"
PY

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
