#!/bin/bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore   # suppress "running as root" pip warning
# =============================================================================
# POD SETUP — the only script you ever run on a pod
#
# Usage:  bash pod_setup.sh
#   (or curl from raw URL and pipe to bash — works either way)
#
# What it does:
#   1. Clones/syncs repo to the 'test' branch
#   2. Installs deps (pip, zstandard, FA3, dataset)
#   3. Verifies everything works
#   4. Done. You run your experiment manually.
# =============================================================================

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="TEST_LAB"
PIP_TORCH_INDEX_URL="${PIP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
REQUIRED_CUDA_PREFIX="${REQUIRED_CUDA_PREFIX:-12.4}"
# Pinned for the known-good 8xH100 stack.
REQUIRED_TORCH_PKGS="${REQUIRED_TORCH_PKGS:-torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1}"
REQUIRED_TORCH_VERSION="${REQUIRED_TORCH_VERSION:-2.4.1+cu124}"
# Auto-detect repo root from script location; fall back for curl-pipe scenario
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)" || true
_CANDIDATE="$(cd -- "${_SCRIPT_DIR}/.." && pwd 2>/dev/null)" || true
if [[ -d "${_CANDIDATE}/.git" ]]; then
    WORKSPACE="${_CANDIDATE}"
else
    WORKSPACE="/workspace/parameter-golf"
fi

echo "============================================"
echo "  POD SETUP"
echo "  Branch: ${BRANCH}"
echo "============================================"

# =============================================================================
# 1. Get the repo on the test branch
# =============================================================================
if [ -d "${WORKSPACE}/.git" ]; then
    echo "[1/6] Repo exists, force-syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    git fetch origin "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
    git clean -fd --quiet
elif [ -d "${WORKSPACE}" ]; then
    echo "[1/6] Existing non-git workspace detected, using in-place files..."
    cd "${WORKSPACE}"
else
    echo "[1/6] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
if [ -d "${WORKSPACE}/.git" ]; then
    echo "  HEAD: $(git log --oneline -1)"
else
    echo "  HEAD: non-git workspace (no commit metadata)"
fi

# =============================================================================
# 2. Verify/repair base environment (system Python + exact SOTA PyTorch)
# =============================================================================
echo ""
echo "[2/6] Checking base environment (must be cu124)..."

python3 --version || { echo "FATAL: python3 not found"; exit 1; }

torch_state() {
python3 - <<PY
import sys
required = "${REQUIRED_CUDA_PREFIX}"
required_torch = "${REQUIRED_TORCH_VERSION}"
try:
    import torch
except Exception as e:
    print(f"  PyTorch import failed: {type(e).__name__}: {e}")
    raise SystemExit(10)
cuda = torch.version.cuda or "NONE"
print(f"  PyTorch {torch.__version__}  CUDA {cuda}")
if not cuda.startswith(required):
    raise SystemExit(20)
if torch.__version__ != required_torch:
    raise SystemExit(21)
PY
}

if torch_state; then
    echo "  Base torch stack OK"
else
    rc=$?
    if [ "${rc}" -eq 10 ]; then
        echo "  PyTorch missing/broken in system Python; installing pinned cu124 stack..."
    elif [ "${rc}" -eq 20 ]; then
        echo "  Wrong CUDA backend detected; reinstalling pinned cu124 stack..."
    elif [ "${rc}" -eq 21 ]; then
        echo "  Wrong torch version detected; reinstalling pinned SOTA torch stack..."
    else
        echo "  Unexpected torch check failure (rc=${rc}); reinstalling pinned cu124 stack..."
    fi
    python3 -m pip install --upgrade pip >/dev/null
    # Clean old torch family first to avoid stale binary mixes.
    python3 -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true
    python3 -m pip install --no-cache-dir --force-reinstall \
        --index-url "${PIP_TORCH_INDEX_URL}" ${REQUIRED_TORCH_PKGS}
    torch_state || { echo "FATAL: torch stack still invalid after reinstall"; exit 1; }
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "  WARNING: No GPUs detected"
else
    python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_mem // 1024**3}GB)')
" 2>/dev/null || true
fi

# =============================================================================
# 3. Core pip packages (system site-packages, no conda, no PYTHONPATH)
# =============================================================================
echo ""
echo "[3/6] Installing pip packages..."

python3 -m pip install --upgrade pip -q 2>&1 | tail -1

python3 -m pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece attr -q 2>&1 | tail -1
echo "  Core packages OK"

# =============================================================================
# 4. zstandard (CRITICAL: prevents artifact size inflation)
# =============================================================================
echo ""
echo "[4/6] zstandard..."

if python3 -c "import zstandard" 2>/dev/null; then
    echo "  Already installed"
else
    python3 -m pip install zstandard -q
    echo "  Installed"
fi
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__}')"

# =============================================================================
# 5. FlashAttention-3
# =============================================================================
echo ""
echo "[5/6] FlashAttention-3..."

install_fa3() {
    echo "  Attempting FA3 abi3 wheel (cu124)..."
    if python3 -m pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi
    echo "  ERROR: Could not install FA3 cu124 wheel."
    return 1
}

python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null \
    || install_fa3 \
    || { echo "FATAL: FA3 unavailable; refusing non-SOTA fallback stack"; exit 1; }

# =============================================================================
# 6. Dataset (sp1024)
# =============================================================================
echo ""
echo "[6/6] Tokenizer + FineWeb dataset (sp1024)..."

# Tokenizer
TOKENIZER="${WORKSPACE}/data/tokenizers/fineweb_1024_bpe.model"
if [ -f "${TOKENIZER}" ]; then
    echo "  Tokenizer already present"
else
    echo "  Downloading tokenizer..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download sproos/parameter-golf-tokenizers \
            --include "tokenizers/*" --local-dir "${WORKSPACE}/data"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    allow_patterns='tokenizers/*',
    local_dir='${WORKSPACE}/data')
"
    fi
    echo "  Tokenizer downloaded"
fi

# Dataset shards — use nullglob array so unmatched glob = 0, not a crash
shopt -s nullglob
_train=("${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin)
_val=("${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin)
TRAIN_COUNT=${#_train[@]}
VAL_COUNT=${#_val[@]}
shopt -u nullglob

if [ "$TRAIN_COUNT" -ge 10 ]; then
    echo "  Already have $TRAIN_COUNT train / $VAL_COUNT val shards"
else
    echo "  Downloading dataset ($TRAIN_COUNT train shards found, need 10+)..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download sproos/parameter-golf-tokenizers \
            --include "datasets/fineweb10B_sp1024/*" --local-dir "${WORKSPACE}/data"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    allow_patterns='datasets/fineweb10B_sp1024/*',
    local_dir='${WORKSPACE}/data')
"
    fi
    echo "  Dataset downloaded"
fi

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - << 'PYEOF'
import sys, glob

print(f"Python       : {sys.version.split()[0]}")
print(f"Executable   : {sys.executable}")

import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

fa = "NOT FOUND"
try:
    from flash_attn_interface import flash_attn_func
    fa = "flash_attn_interface (FA3 hopper)"
except ImportError:
    try:
        import flash_attn
        v = flash_attn.__version__
        fa = f"flash_attn v{v}" + ("" if v.startswith("3") else " WARNING: not FA3!")
    except ImportError:
        pass
print(f"FlashAttn    : {fa}")

try:
    import zstandard
    print(f"zstandard    : {zstandard.__version__}")
except ImportError:
    print("zstandard    : MISSING!")

try:
    import sentencepiece
    print(f"sentencepiece: OK")
except ImportError:
    print("sentencepiece: MISSING!")

train = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
val   = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
print(f"Train shards : {len(train)}")
print(f"Val shards   : {len(val)}")
PYEOF

echo ""
echo "============================================"
echo " READY."
echo "============================================"
