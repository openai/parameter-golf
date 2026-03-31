#!/bin/bash
set -euo pipefail
# ================================================================
#  BW7 MegaGate — Fresh Pod Setup + Run
#
#  Paste this whole block into the pod terminal. It will:
#    1. Clone repo (TEST_LAB branch)
#    2. Install pip deps
#    3. Install FlashAttention-3
#    4. Download dataset + tokenizer
#    5. Launch the 8-arm ablation
#
#  Usage:
#    SEED=444 NPROC_PER_NODE=4 bash pod_setup.sh
# ================================================================

REPO_URL="https://github.com/newjordan/parameter-golf.git"
BRANCH="TEST_LAB"
WORKSPACE="/workspace/parameter-golf-lab"
SEED="${SEED:-444}"
NPROC="${NPROC_PER_NODE:-4}"

echo "============================================"
echo "  BW7 MegaGate — Pod Setup"
echo "  Branch   : ${BRANCH}"
echo "  Workspace: ${WORKSPACE}"
echo "  Seed     : ${SEED}  GPUs: ${NPROC}"
echo "============================================"

# ----------------------------------------------------------------
# 1. Clone or sync repo
# ----------------------------------------------------------------
echo ""
echo "[1/6] Repo..."
if [ -d "${WORKSPACE}/.git" ]; then
    echo "  Repo exists — force-syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    git fetch origin "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
    git clean -fd --quiet
else
    echo "  Cloning..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
echo "  HEAD: $(git log --oneline -1)"

# ----------------------------------------------------------------
# 2. Verify base environment
# ----------------------------------------------------------------
echo ""
echo "[2/6] Environment..."
python3 --version
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')" \
    || { echo "FATAL: PyTorch not found"; exit 1; }

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  GPUs detected: ${GPU_COUNT}"
python3 - << 'PYEOF'
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory // 1024**3}GB)")
PYEOF

# ----------------------------------------------------------------
# 3. Pip deps
# ----------------------------------------------------------------
echo ""
echo "[3/6] Installing pip packages..."
pip install --upgrade pip -q 2>&1 | tail -1
pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" sentencepiece zstandard -q 2>&1 | tail -1
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')"

# ----------------------------------------------------------------
# 4. FlashAttention-3
# ----------------------------------------------------------------
echo ""
echo "[4/6] FlashAttention-3..."

_fa3_ok() {
    python3 -c "from flash_attn_interface import flash_attn_func; print('  FA3 (flash_attn_interface) OK')" 2>/dev/null && return 0
    python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null && return 0
    return 1
}

if _fa3_ok; then
    echo "  Already installed"
else
    echo "  Trying cu128 wheel..."
    pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        -q 2>&1 | tail -3 || true

    if ! _fa3_ok; then
        echo "  cu128 failed — trying cu124..."
        pip install --no-cache-dir \
            "https://download.pytorch.org/whl/cu124/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
            -q 2>&1 | tail -3 || true
    fi

    if ! _fa3_ok; then
        echo "  Wheels failed — symlinking from repo source..."
        SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
        SRC="${WORKSPACE}/flash-attention/hopper/flash_attn_interface.py"
        if [ -f "${SRC}" ]; then
            ln -sf "${SRC}" "${SITE}/flash_attn_interface.py"
            echo "  Symlinked flash_attn_interface.py into site-packages"
        else
            echo "  WARNING: FA3 not available — will use PyTorch SDPA fallback"
        fi
    fi
fi

# ----------------------------------------------------------------
# 5. Dataset + tokenizer
# ----------------------------------------------------------------
echo ""
echo "[5/6] Dataset (fineweb10B_sp1024)..."

TRAIN_COUNT=$(ls "${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin 2>/dev/null | wc -l || echo 0)
VAL_COUNT=$(ls "${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_val_"*.bin 2>/dev/null | wc -l || echo 0)

if [ "${TRAIN_COUNT}" -ge 10 ]; then
    echo "  Already have ${TRAIN_COUNT} train / ${VAL_COUNT} val shards — skipping download"
else
    echo "  Downloading (found ${TRAIN_COUNT} train shards — need 10+)..."
    python3 - << PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    "sproos/parameter-golf-tokenizers",
    allow_patterns=["datasets/fineweb10B_sp1024/*", "tokenizers/*"],
    local_dir="${WORKSPACE}/data",
)
print("  Download complete")
PYEOF
fi

TRAIN_COUNT=$(ls "${WORKSPACE}/data/datasets/fineweb10B_sp1024/fineweb_train_"*.bin 2>/dev/null | wc -l || echo 0)
[[ -f "${WORKSPACE}/data/tokenizers/fineweb_1024_bpe.model" ]] && TOK_OK="OK" || TOK_OK="MISSING"
echo "  train shards: ${TRAIN_COUNT}  tokenizer: ${TOK_OK}"

# ----------------------------------------------------------------
# 6. Run ablation
# ----------------------------------------------------------------
echo ""
echo "[6/6] Launching BW7 MegaGate ablation..."
echo "  seed=${SEED}  GPUs=${NPROC}"
echo ""

cd "${WORKSPACE}"
SEED="${SEED}" NPROC_PER_NODE="${NPROC}" \
    bash crawler/2026-03-31_BW7_MegaGate/run_ablation.sh
