#!/bin/bash
# -------------------------------------------------------------------------------
# Parameter Golf -- Complete Environment Setup Script
# Drop this into the project root and run: bash setup.sh
# -------------------------------------------------------------------------------

set -e

echo "----------------------------------------------"
echo " Parameter Golf -- Environment Setup"
echo "----------------------------------------------"

# -------------------------------------------------------------------------------
# 1. Miniconda
# -------------------------------------------------------------------------------
echo ""
echo "[1/5] Miniconda..."

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
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
echo "    TOS accepted."

# -------------------------------------------------------------------------------
# 2. Python Environment
# -------------------------------------------------------------------------------
echo ""
echo "[2/5] Python 3.13 environment..."

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
echo "[3/5] Requirements..."

if python3 -c "import torch, sentencepiece, numpy" 2>/dev/null; then
    echo "    Core packages already installed -- skipping."
else
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 4. FlashAttention-3
# -------------------------------------------------------------------------------
echo ""
echo "[4/5] FlashAttention-3..."

if python3 -c "import flash_attn" 2>/dev/null || python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo "    Already installed -- skipping."
else
    # abi3 wheel -- Python 3.9+ compatible, installs in seconds, no compilation
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 5. Dataset
# -------------------------------------------------------------------------------
echo ""
echo "[5/5] FineWeb dataset (sp8192, 10 shards)..."

echo "    Downloading dataset..."
hf download sproos/parameter-golf-tokenizers --include "datasets/fineweb10B_sp8192/*" --local-dir ./data

# Ensure tokenizer is available (bundled in submission folder as fallback)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ./data/tokenizers
if [ ! -f ./data/tokenizers/fineweb_8192_bpe.model ]; then
    if [ -f "$SCRIPT_DIR/fineweb_8192_bpe.model" ]; then
        cp "$SCRIPT_DIR/fineweb_8192_bpe.model" ./data/tokenizers/
        cp "$SCRIPT_DIR/fineweb_8192_bpe.vocab" ./data/tokenizers/ 2>/dev/null || true
        echo "    Tokenizer copied from submission folder."
    else
        python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 0 2>/dev/null || true
        echo "    Tokenizer downloaded via cached_challenge_fineweb."
    fi
fi
echo "    Downloaded."

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
        print(f"  GPU {i}      : {props.name} ({props.total_memory // 1024**3}GB)")

try:
    import flash_attn
    print(f"FlashAttn    : {flash_attn.__version__}")
except ImportError:
    try:
        import flash_attn_interface
        print(f"FlashAttn3   : available")
    except ImportError:
        print(f"FlashAttn    : NOT found")

train_files = sorted(glob.glob("./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin"))
val_files   = sorted(glob.glob("./data/datasets/fineweb10B_sp8192/fineweb_val_*.bin"))
print(f"Train shards : {len(train_files)}")
print(f"Val shards   : {len(val_files)}")

if val_files:
    total = sum(
        int(np.fromfile(f, dtype='<i4', count=3)[2])
        for f in val_files
    )
    print(f"Val tokens   : {total:,}")
EOF

echo ""
echo "----------------------------------------------"
echo " Done. Run training with:"
echo "   conda activate golf"
echo "   bash run_cuda_feedback.sh"
echo "----------------------------------------------"
