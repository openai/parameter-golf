#!/bin/bash
# -------------------------------------------------------------------------------
# Parameter Golf -- Pod Setup (RunPod / Vast.ai)
# Uses the DEFAULT system Python + PyTorch. No conda. No PYTHONPATH hacks.
#
# Run once after pod starts:
#   bash experiments/setup_runpod.sh
# -------------------------------------------------------------------------------

set -euo pipefail

echo "============================================"
echo " Parameter Golf -- Pod Environment Setup"
echo "============================================"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------------------------
# 1. Verify base environment (system Python + PyTorch must already exist)
# -------------------------------------------------------------------------------
echo ""
echo "[1/5] Checking base environment..."

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

# -------------------------------------------------------------------------------
# 2. Core pip packages (into system site-packages, no conda)
# -------------------------------------------------------------------------------
echo ""
echo "[2/5] Installing pip packages..."

pip install --upgrade pip -q 2>&1 | tail -1

# Install requirements but skip torch (already installed by the pod image)
pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece -q 2>&1 | tail -1
echo "  Core packages OK"

# -------------------------------------------------------------------------------
# 3. zstandard (CRITICAL: prevents artifact size inflation)
# -------------------------------------------------------------------------------
echo ""
echo "[3/5] zstandard..."

if python3 -c "import zstandard" 2>/dev/null; then
    echo "  Already installed"
else
    pip install zstandard -q
    echo "  Installed"
fi
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__}')"

# -------------------------------------------------------------------------------
# 4. FlashAttention-3 (into system site-packages -- no PYTHONPATH needed)
# -------------------------------------------------------------------------------
echo ""
echo "[4/5] FlashAttention-3..."

install_fa3() {
    echo "  Attempting FA3 abi3 wheel..."
    if pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi

    echo "  abi3 wheel failed, trying cu124..."
    if pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>&1 | tail -3; then
        return 0
    fi

    echo "  Wheels failed. Checking for local flash-attention/hopper source..."
    if [ -d "${REPO_ROOT}/flash-attention/hopper" ]; then
        # Symlink the hopper interface into site-packages so it's always importable
        SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
        SRC="${REPO_ROOT}/flash-attention/hopper/flash_attn_interface.py"
        if [ -f "$SRC" ]; then
            ln -sf "$SRC" "${SITE}/flash_attn_interface.py"
            echo "  Symlinked flash_attn_interface.py into site-packages"
            return 0
        fi
    fi

    echo "  WARNING: Could not install FA3. Will fall back to PyTorch SDPA."
    return 1
}

# Check if FA3 already works
if python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null; then
    : # already good
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null; then
    : # flash_attn v3 package works
else
    install_fa3
fi

# -------------------------------------------------------------------------------
# 5. Dataset (sp1024)
# -------------------------------------------------------------------------------
echo ""
echo "[5/5] FineWeb dataset (sp1024)..."

TRAIN_COUNT=$(ls "${REPO_ROOT}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls "${REPO_ROOT}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -ge 10 ]; then
    echo "  Already have $TRAIN_COUNT train / $VAL_COUNT val shards"
else
    echo "  Downloading ($TRAIN_COUNT train shards found, need 10+)..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download sproos/parameter-golf-tokenizers \
            --include "datasets/fineweb10B_sp1024/*" --local-dir "${REPO_ROOT}/data"
    else
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    allow_patterns='datasets/fineweb10B_sp1024/*',
    local_dir='${REPO_ROOT}/data')
"
    fi
    echo "  Downloaded"
fi

# -------------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Verification"
echo "============================================"

python3 - << 'PYEOF'
import sys, os

print(f"Python       : {sys.version.split()[0]}")
print(f"Executable   : {sys.executable}")

import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

# FA3
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

# zstandard
try:
    import zstandard
    print(f"zstandard    : {zstandard.__version__}")
except ImportError:
    print("zstandard    : MISSING!")

# sentencepiece
try:
    import sentencepiece
    print(f"sentencepiece: OK")
except ImportError:
    print("sentencepiece: MISSING!")

import glob
train = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))
val   = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
print(f"Train shards : {len(train)}")
print(f"Val shards   : {len(val)}")
PYEOF

echo ""
echo "============================================"
echo " Setup complete. No conda needed."
echo " Just run your experiment directly:"
echo "   bash experiments/A_wing/green_2/run.sh"
echo "============================================"
