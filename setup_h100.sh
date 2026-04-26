#!/bin/bash
# ---------------------------------------------------------------------------
# Parameter Golf v2 -- H100 Environment Setup
# Run from project root: bash setup_h100.sh
# ---------------------------------------------------------------------------
set -e

echo "============================================"
echo " Parameter Golf v2 -- H100 Setup"
echo "============================================"

# 1. Miniconda
echo "[1/5] Miniconda..."
if [ -d "$HOME/miniconda3" ]; then
    echo "  Already installed."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b
    rm /tmp/miniconda.sh
    ~/miniconda3/bin/conda init bash
fi
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

# 2. Python environment
echo "[2/5] Python 3.13 environment..."
if conda env list | grep -q "^golf "; then
    echo "  Already exists."
else
    conda create -n golf python=3.13 -y
fi
conda activate golf

# 3. Requirements
echo "[3/5] Python packages..."
pip install --upgrade pip -q
pip install numpy torch sentencepiece brotli -q

# 4. FlashAttention-3 (Hopper)
echo "[4/5] FlashAttention-3..."
if python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo "  Already installed."
else
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
fi

# 5. Dataset
echo "[5/5] FineWeb dataset (sp8192)..."
SHARD_COUNT=$(ls ./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -ge 10 ]; then
    echo "  Already downloaded ($SHARD_COUNT shards)."
else
    pip install huggingface-hub -q
    huggingface-cli download sproos/parameter-golf-tokenizers --include "datasets/fineweb10B_sp8192/*" --local-dir ./data
fi

# Verify
echo ""
echo "============================================"
echo " Verification"
echo "============================================"
python3 - << 'PYEOF'
import sys, torch, glob
print(f"Python     : {sys.version.split()[0]}")
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA       : {torch.cuda.is_available()}")
print(f"GPUs       : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}    : {p.name} ({p.total_mem // 1024**3}GB)")
try:
    import flash_attn_interface; print("FlashAttn3 : available")
except ImportError:
    try:
        import flash_attn; print(f"FlashAttn2 : {flash_attn.__version__}")
    except ImportError:
        print("FlashAttn  : NOT found")
import brotli; print("Brotli     : available")
trains = glob.glob("./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin")
vals = glob.glob("./data/datasets/fineweb10B_sp8192/fineweb_val_*.bin")
print(f"Train shards: {len(trains)}")
print(f"Val shards  : {len(vals)}")
PYEOF

echo ""
echo "Done. Run: conda activate golf && bash run_h100.sh"
