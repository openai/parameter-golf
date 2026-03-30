#!/bin/bash
set -euo pipefail
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
BRANCH="test"
WORKSPACE="/workspace/parameter-golf"

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
else
    echo "[1/6] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
echo "  HEAD: $(git log --oneline -1)"

# =============================================================================
# 2. Verify base environment (system Python + PyTorch must already exist)
# =============================================================================
echo ""
echo "[2/6] Checking base environment..."

python3 --version || { echo "FATAL: python3 not found"; exit 1; }
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')" \
    || { echo "FATAL: PyTorch not installed in system Python"; exit 1; }

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

pip install --upgrade pip -q 2>&1 | tail -1

pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece -q 2>&1 | tail -1
echo "  Core packages OK"

# =============================================================================
# 4. zstandard (CRITICAL: prevents artifact size inflation)
# =============================================================================
echo ""
echo "[4/6] zstandard..."

if python3 -c "import zstandard" 2>/dev/null; then
    echo "  Already installed"
else
    pip install zstandard -q
    echo "  Installed"
fi
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__}')"

# =============================================================================
# 5. FlashAttention-3
# =============================================================================
echo ""
echo "[5/6] FlashAttention-3..."

install_fa3() {
    echo "  Attempting FA3 abi3 wheel (cu128)..."
    if pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi

    echo "  cu128 failed, trying cu124..."
    if pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi

    echo "  Wheels failed. Checking for local flash-attention/hopper source..."
    if [ -d "${WORKSPACE}/flash-attention/hopper" ]; then
        SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
        SRC="${WORKSPACE}/flash-attention/hopper/flash_attn_interface.py"
        if [ -f "$SRC" ]; then
            ln -sf "$SRC" "${SITE}/flash_attn_interface.py"
            echo "  Symlinked flash_attn_interface.py into site-packages"
            return 0
        fi
    fi

    echo "  WARNING: Could not install FA3. Will fall back to PyTorch SDPA."
    return 1
}

if python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null; then
    : # already good
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null; then
    : # flash_attn v3 package works
else
    install_fa3
fi

# =============================================================================
# 6. Dataset (sp1024)
# =============================================================================
echo ""
echo "[6/6] FineWeb dataset (sp1024)..."

TRAIN_COUNT=$(ls "${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls "${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -ge 10 ]; then
    echo "  Already have $TRAIN_COUNT train / $VAL_COUNT val shards"
else
    echo "  Downloading ($TRAIN_COUNT train shards found, need 10+)..."
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
    echo "  Downloaded"
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
