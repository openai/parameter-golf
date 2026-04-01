#!/bin/bash
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore
# =============================================================================
# POD SETUP — force the exact SOTA stack, every time, no exceptions.
#
# Usage:  bash scripts/pod_setup.sh          (from repo root)
#    or:  bash pod_setup.sh                  (from scripts/)
#
# What it does:
#   1. Syncs repo to TEST_LAB
#   2. Forces torch 2.4.1+cu124 (overwrites whatever the pod has)
#   3. Pins sympy 1.12.1 (1.13+ breaks inductor on CUDA 13.0 drivers)
#   4. Installs FA3 for cu124+torch2.4
#   5. Installs pip deps + zstandard
#   6. Downloads dataset
#   7. Verifies everything — FAILS HARD if wrong
#
# The CUDA *driver* can be anything >= 12.4 (forward compat is fine).
# The torch *runtime* MUST be exactly 2.4.1+cu124 for SOTA reproducibility.
# Different torch versions produce different weight distributions which
# change int6+zstd compression ratios by >1MB. This is not optional.
# =============================================================================

# ── Pinned versions (do NOT change without a full A/B verification) ──
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
TORCH_PKGS="torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
TORCH_EXPECTED="2.4.1+cu124"
CUDA_EXPECTED_PREFIX="12.4"
SYMPY_PIN="1.12.1"

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="TEST_LAB"

die() { echo ""; echo "FATAL: $*" >&2; exit 1; }

# ── Detect workspace ──
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd 2>/dev/null)" || true
_CANDIDATE="$(cd -- "${_SCRIPT_DIR}/.." && pwd 2>/dev/null)" || true
if [[ -d "${_CANDIDATE}/.git" ]]; then
    WORKSPACE="${_CANDIDATE}"
else
    WORKSPACE="/workspace/parameter-golf"
fi

echo "============================================"
echo "  POD SETUP — SOTA stack (cu124)"
echo "  Branch: ${BRANCH}"
echo "============================================"

# =============================================================================
# 1. Get the repo on TEST_LAB
# =============================================================================
if [ -d "${WORKSPACE}/.git" ]; then
    echo "[1/7] Repo exists — syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    git fetch origin "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
    git clean -fd --quiet
elif [ -d "${WORKSPACE}" ]; then
    echo "[1/7] Existing non-git workspace — using in-place..."
    cd "${WORKSPACE}"
else
    echo "[1/7] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
[ -d .git ] && echo "  HEAD: $(git log --oneline -1)" || echo "  (non-git workspace)"

# =============================================================================
# 2. Force torch 2.4.1+cu124 — always, unconditionally
# =============================================================================
echo ""
echo "[2/7] Forcing torch ${TORCH_EXPECTED}..."

python3 --version || die "python3 not found"

# Check if already correct (skip reinstall to save time)
_current_torch=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")
if [[ "${_current_torch}" == "${TORCH_EXPECTED}" ]]; then
    echo "  Already installed: ${_current_torch}"
else
    echo "  Current: ${_current_torch} — replacing..."
    python3 -m pip install --upgrade pip -q 2>&1 | tail -1
    # Purge old torch family to avoid stale binary mixes
    python3 -m pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true
    python3 -m pip install --no-cache-dir --force-reinstall \
        --index-url "${TORCH_INDEX}" ${TORCH_PKGS}
fi

# Hard verify — die if wrong
_verify_torch=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
_verify_cuda=$(python3 -c "import torch; print(torch.version.cuda or 'NONE')" 2>/dev/null)
[[ "${_verify_torch}" == "${TORCH_EXPECTED}" ]] || \
    die "torch install failed: got '${_verify_torch}', need '${TORCH_EXPECTED}'"
[[ "${_verify_cuda}" == "${CUDA_EXPECTED_PREFIX}"* ]] || \
    die "CUDA mismatch: got '${_verify_cuda}', need '${CUDA_EXPECTED_PREFIX}x'"
echo "  torch=${_verify_torch}  cuda=${_verify_cuda}  OK"

# GPU check
python3 -c "
import torch
n = torch.cuda.device_count()
if n == 0:
    print('  WARNING: No GPUs detected')
else:
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory // 1024**3}GB)')
" 2>/dev/null || true

# =============================================================================
# 3. Pin sympy (1.13+ has NaN comparison bug in inductor bounds)
# =============================================================================
echo ""
echo "[3/7] Pinning sympy==${SYMPY_PIN}..."

_current_sympy=$(python3 -c "import sympy; print(sympy.__version__)" 2>/dev/null || echo "NONE")
if [[ "${_current_sympy}" == "${SYMPY_PIN}" ]]; then
    echo "  Already correct: ${_current_sympy}"
else
    echo "  Current: ${_current_sympy} — replacing..."
    python3 -m pip install "sympy==${SYMPY_PIN}" -q
fi
python3 -c "import sympy; assert sympy.__version__ == '${SYMPY_PIN}', f'got {sympy.__version__}'" \
    || die "sympy pin failed"
echo "  sympy=${SYMPY_PIN}  OK"

# =============================================================================
# 4. Core pip packages + zstandard
# =============================================================================
echo ""
echo "[4/7] Core pip packages..."

python3 -m pip install numpy tqdm huggingface-hub kernels setuptools wheel \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece attr zstandard -q 2>&1 | tail -1
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__}')"
echo "  Core packages OK"

# =============================================================================
# 5. FlashAttention-3 for cu124+torch2.4
# =============================================================================
echo ""
echo "[5/7] FlashAttention-3..."

# Always start clean — purge any FA3 built against a different torch
python3 -m pip uninstall -y flash-attn flash_attn flash_attn_3 2>/dev/null || true

_pyver=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
_fa3_installed=0

# Try GitHub release wheels (cu124+torch2.4, both ABI variants)
for _abi in FALSE TRUE; do
    _url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu124torch2.4cxx11abi${_abi}-${_pyver}-${_pyver}-linux_x86_64.whl"
    echo "  Trying: flash_attn-2.7.4 cu124 cxx11abi${_abi} (${_pyver})..."
    if python3 -m pip install --no-cache-dir "${_url}" -q 2>&1 | tail -2; then
        if python3 -c "import flash_attn" 2>/dev/null; then
            _fa3_installed=1
            break
        fi
    fi
done

# Fallback: build from source if no pre-built wheel for this Python version
if [[ "${_fa3_installed}" -eq 0 ]]; then
    echo "  No pre-built wheel for ${_pyver} — building from source (~5-10 min)..."
    python3 -m pip install --no-cache-dir flash-attn --no-build-isolation -q 2>&1 | tail -5 \
        && _fa3_installed=1
fi

# Symlink flash_attn_interface into site-packages if needed
# (training script does "from flash_attn_interface import ..." at top level)
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    _iface=$(python3 -c "
import os, importlib.util
for pkg in ('flash_attn', 'flash_attn_3'):
    spec = importlib.util.find_spec(pkg)
    if spec and spec.origin:
        p = os.path.join(os.path.dirname(spec.origin), 'flash_attn_interface.py')
        if os.path.exists(p):
            print(p)
            break
" 2>/dev/null || true)
    if [[ -n "${_iface}" ]]; then
        _site=$(python3 -c "import site; print(site.getsitepackages()[0])")
        ln -sf "${_iface}" "${_site}/flash_attn_interface.py"
        echo "  Symlinked flash_attn_interface.py → site-packages"
    fi
fi

# Hard verify FA3
python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" \
    || die "FA3 import failed. Wheel may not exist for ${_pyver}. Check https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4"

# =============================================================================
# 6. Dataset (sp1024)
# =============================================================================
echo ""
echo "[6/7] Tokenizer + FineWeb dataset (sp1024)..."

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
# 7. Final verification — hard fail if anything is wrong
# =============================================================================
echo ""
echo "============================================"
echo "  VERIFICATION"
echo "============================================"

python3 - <<'PYEOF'
import sys, glob

print(f"Python       : {sys.version.split()[0]}")
print(f"Executable   : {sys.executable}")

import torch
tv = torch.__version__
cv = torch.version.cuda or "NONE"
print(f"PyTorch      : {tv}")
print(f"CUDA runtime : {cv}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

assert tv == "2.4.1+cu124", f"WRONG TORCH: {tv}"
assert cv.startswith("12.4"),  f"WRONG CUDA: {cv}"

import sympy
sv = sympy.__version__
print(f"sympy        : {sv}")
assert sv == "1.12.1", f"WRONG SYMPY: {sv}"

from flash_attn_interface import flash_attn_func
print(f"FlashAttn    : FA3 (flash_attn_interface) OK")

import zstandard
print(f"zstandard    : {zstandard.__version__}")

import sentencepiece
print(f"sentencepiece: OK")

train = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
val   = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
print(f"Train shards : {len(train)}")
print(f"Val shards   : {len(val)}")

assert len(train) >= 10, f"Only {len(train)} train shards"
assert len(val)   >= 1,  f"No val shards"

print()
print("ALL CHECKS PASSED")
PYEOF

echo ""
echo "============================================"
echo "  READY — torch=${TORCH_EXPECTED} sympy=${SYMPY_PIN}"
echo "============================================"
