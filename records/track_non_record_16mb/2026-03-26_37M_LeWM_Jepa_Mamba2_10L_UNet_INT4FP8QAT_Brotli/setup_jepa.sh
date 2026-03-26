#!/bin/bash
# -------------------------------------------------------------------------------
# To JEPA Or Not To JEPA - Complete Environment Setup Script
# Drop this into the project root and run: bash setup.sh
# -------------------------------------------------------------------------------

set -e

echo "----------------------------------------------"
echo " To JEPA Or Not To JEPA -- Environment Setup"
echo "----------------------------------------------"

# -------------------------------------------------------------------------------
# 1. Miniconda
# -------------------------------------------------------------------------------
echo ""
echo "[1/6] Miniconda..."

if [ -d "$HOME/miniconda3" ]; then
    echo "    Already installed -- skipping."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b
    rm /tmp/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo "    Installed."
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

echo "    Accepting conda TOS..."
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
echo "    TOS accepted."

# -------------------------------------------------------------------------------
# 2. Python Environment
# -------------------------------------------------------------------------------
echo ""
echo "[2/6] Python 3.13 environment..."

if conda env list | grep -q "^golf "; then
    echo "    Environment 'golf' already exists -- skipping."
else
    conda create -n golf python=3.13 -y
    echo "    Created."
fi

conda activate golf
echo "    Activated."

# -------------------------------------------------------------------------------
# 3. Requirements
# -------------------------------------------------------------------------------
echo ""
echo "[3/6] Requirements..."

if python3 -c "import torch, numpy" 2>/dev/null; then
    echo "    Core packages already installed -- skipping."
else
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 4. Mamba-2 SSM (CUDA kernels)
# -------------------------------------------------------------------------------
echo ""
echo "[4/6] Mamba-2 SSM..."

if python3 -c "from mamba_ssm import Mamba2; print('OK')" 2>/dev/null; then
    echo "    Already installed -- skipping."
else
    echo "    Installing causal-conv1d (compiling CUDA kernels, ~5 min)..."
    # Disable CUDA version check if system CUDA != PyTorch CUDA
    TORCH_EXT=$(python3 -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.__file__)")
    sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass  # raise RuntimeError(CUDA_MISMATCH_MESSAGE/' "$TORCH_EXT"

    pip install causal-conv1d --no-build-isolation --no-cache-dir
    echo "    causal-conv1d installed."

    echo "    Installing mamba-ssm (compiling CUDA kernels, ~10 min)..."
    pip install mamba-ssm --no-build-isolation --no-cache-dir
    echo "    mamba-ssm installed."

    # Restore CUDA version check
    sed -i 's/pass  # raise RuntimeError(CUDA_MISMATCH_MESSAGE/raise RuntimeError(CUDA_MISMATCH_MESSAGE/' "$TORCH_EXT"

    # Verify
    python3 -c "from mamba_ssm import Mamba2; print('    Mamba2 verified OK')"
fi

# -------------------------------------------------------------------------------
# 5. FlashAttention-3 (optional, not used by JEPA but useful for baselines)
# -------------------------------------------------------------------------------
echo ""
echo "[5/6] FlashAttention-3..."

if python3 -c "import flash_attn" 2>/dev/null || python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo "    Already installed -- skipping."
else
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" 2>/dev/null || \
        echo "    FlashAttention-3 install failed (not critical for JEPA)."
fi

# -------------------------------------------------------------------------------
# 6. Dataset -- FineWeb bytes (raw UTF-8, no tokenizer needed)
# -------------------------------------------------------------------------------
echo ""
echo "[6/6] FineWeb byte-level dataset..."

BYTES_DIR="./data/datasets/fineweb_bytes_clean"
if [ -d "$BYTES_DIR" ] && ls "$BYTES_DIR"/fineweb_val_*.bin 1>/dev/null 2>&1; then
    TRAIN_COUNT=$(ls "$BYTES_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
    VAL_COUNT=$(ls "$BYTES_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
    echo "    Already downloaded ($TRAIN_COUNT train, $VAL_COUNT val shards) -- skipping."
else
    echo "    Downloading byte-level FineWeb (all shards)..."
    hf download mistobaan/fineweb10B_bytes --repo-type dataset --local-dir "$BYTES_DIR"
    echo "    Downloaded."
fi

# -------------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------"
echo " Verification"
echo "----------------------------------------------"

python3 - << 'EOF'
import sys
import torch
import numpy as np
import glob

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {props.name} ({props.total_mem // 1024**3}GB)")

# Mamba-2
try:
    from mamba_ssm import Mamba2
    print(f"Mamba-2      : available")
except ImportError:
    print(f"Mamba-2      : NOT found")

# FlashAttention
try:
    import flash_attn
    print(f"FlashAttn    : {flash_attn.__version__}")
except ImportError:
    try:
        import flash_attn_interface
        print(f"FlashAttn3   : available")
    except ImportError:
        print(f"FlashAttn    : NOT found (not needed for JEPA)")

# Byte dataset
bytes_dir = "./data/datasets/fineweb_bytes_clean"
train_files = sorted(glob.glob(f"{bytes_dir}/fineweb_train_*.bin"))
val_files   = sorted(glob.glob(f"{bytes_dir}/fineweb_val_*.bin"))
print(f"Train shards : {len(train_files)}")
print(f"Val shards   : {len(val_files)}")

if val_files:
    total = sum(
        int(np.fromfile(f, dtype='<i4', count=3)[2])
        for f in val_files
    )
    print(f"Val bytes    : {total:,}")

if train_files:
    total = sum(
        int(np.fromfile(f, dtype='<i4', count=3)[2])
        for f in train_files[:3]
    )
    avg = total / min(3, len(train_files))
    est = avg * len(train_files)
    print(f"Train bytes  : ~{est:,.0f} (estimated)")
EOF

echo ""
echo "----------------------------------------------"
echo " Done. Run training with:"
echo "   conda activate golf"
echo "   torchrun --standalone --nproc_per_node=8 train_jepa_ssm.py"
echo "----------------------------------------------"
