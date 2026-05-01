#!/bin/bash
# -------------------------------------------------------------------------------
# Parameter Golf -- Quinary submission environment setup
# Run from the submission/ directory on a fresh 8xH100 pod.
#
# After this finishes:
#   - lrzip is installed (used by per-stream compression)
#   - Python deps from requirements.txt are installed
#   - FlashAttention-3 wheel is installed (Hopper-only)
#   - sp16384 tokenizer + tokenized FineWeb shards are at ./data/
#
# Total time on a fresh pod: ~10-25 min (mostly the ~23 GB HF download).
# -------------------------------------------------------------------------------

set -e

echo "=============================================="
echo " Parameter Golf -- Quinary submission setup"
echo "=============================================="

# --------------------------------------------------------------------
# 1. System packages (lrzip; needed by per-stream artifact compression)
# --------------------------------------------------------------------
echo ""
echo "[1/4] System packages (lrzip)..."

if command -v lrzip >/dev/null 2>&1; then
    echo "    lrzip already installed -- skipping."
else
    apt-get update -qq
    apt-get install -y -qq lrzip
    echo "    Installed."
fi

# --------------------------------------------------------------------
# 2. Python requirements
# --------------------------------------------------------------------
echo ""
echo "[2/4] Python requirements..."

if python3 -c "import torch, sentencepiece, numpy, huggingface_hub" 2>/dev/null; then
    echo "    Core packages already installed -- skipping."
else
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "    Installed."
fi

# --------------------------------------------------------------------
# 3. FlashAttention-3 (Hopper-specific wheel)
# --------------------------------------------------------------------
echo ""
echo "[3/4] FlashAttention-3..."

if python3 -c "import flash_attn_interface" 2>/dev/null; then
    echo "    Already installed -- skipping."
else
    pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    echo "    Installed."
fi

# --------------------------------------------------------------------
# 4. FineWeb dataset + sp16384 tokenizer (canonical/ subset only)
# --------------------------------------------------------------------
echo ""
echo "[4/4] FineWeb sp16384 dataset + tokenizer..."

if ls ./data/canonical/datasets/fineweb10B_sp16384/fineweb_val_*.bin 1>/dev/null 2>&1; then
    echo "    Already present at ./data/canonical/ -- skipping."
else
    echo "    Downloading from deniskurlov/parameter-golf-fineweb-sp16384 (canonical/ only, ~23 GB)..."
    hf download deniskurlov/parameter-golf-fineweb-sp16384 \
        --include "canonical/**" \
        --local-dir ./data \
        --repo-type dataset
    echo "    Downloaded."
fi

# --------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Verification"
echo "=============================================="

python3 - << 'EOF'
import sys, glob
import torch, numpy as np

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}      : {props.name} ({props.total_memory // 1024**3} GB)")

try:
    import flash_attn_interface  # noqa
    print("FlashAttn3   : installed")
except ImportError:
    print("FlashAttn3   : NOT found (required for training)")

import sentencepiece as spm
sp_path = "./data/canonical/tokenizers/fineweb_16384_bpe.model"
sp = spm.SentencePieceProcessor(model_file=sp_path)
print(f"Tokenizer    : {sp.vocab_size()}-vocab SentencePiece BPE @ {sp_path}")

train = sorted(glob.glob("./data/canonical/datasets/fineweb10B_sp16384/fineweb_train_*.bin"))
val   = sorted(glob.glob("./data/canonical/datasets/fineweb10B_sp16384/fineweb_val_*.bin"))
total_val = sum(int(np.fromfile(f, dtype="<i4", count=3)[2]) for f in val) if val else 0
print(f"Dataset      : {len(train)} train shards, {len(val)} val shards, {total_val:,} val tokens")

import shutil
print(f"lrzip binary : {shutil.which('lrzip') or 'NOT FOUND (required for per-stream compression)'}")
EOF

echo ""
echo "=============================================="
echo " Done. To train + evaluate:"
echo ""
echo "   bash run.sh"
echo ""
echo " Or with overrides (e.g., a different seed):"
echo ""
echo "   SEED=1337 bash run.sh"
echo "=============================================="
