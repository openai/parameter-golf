#!/bin/bash
set -euo pipefail
# =============================================================================
# COBRA POD SETUP — setup focused on base-quality harness workflow
#
# Usage:
#   bash experiments/pod_setup_cobra.sh
#
# What it does:
#   1. Clones/syncs repo to the test branch
#   2. Installs deps (pip, zstandard, FA3, dataset)
#   3. Verifies Cobra harness files and prints racecar commands
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/newjordan/parameter-golf.git}"
BRANCH="${BRANCH:-test}"
WORKSPACE="${WORKSPACE:-/workspace/parameter-golf-lab}"

echo "============================================"
echo "  COBRA POD SETUP"
echo "  Branch   : ${BRANCH}"
echo "  Workspace: ${WORKSPACE}"
echo "============================================"

# =============================================================================
# 1. Get the repo on the target branch
# =============================================================================
if [ -d "${WORKSPACE}/.git" ]; then
    echo "[1/7] Repo exists, force-syncing to ${BRANCH}..."
    cd "${WORKSPACE}"
    git fetch origin "${BRANCH}" --quiet
    git checkout -B "${BRANCH}" "origin/${BRANCH}" --force
    git clean -fd --quiet
else
    echo "[1/7] Cloning repo..."
    git clone -b "${BRANCH}" "${REPO_URL}" "${WORKSPACE}"
    cd "${WORKSPACE}"
fi
echo "  HEAD: $(git log --oneline -1)"

# =============================================================================
# 2. Verify base environment
# =============================================================================
echo ""
echo "[2/7] Checking base environment..."

python3 --version || { echo "FATAL: python3 not found"; exit 1; }
python3 -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA {torch.version.cuda}')" \
    || { echo "FATAL: PyTorch not installed in system Python"; exit 1; }

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "  WARNING: No GPUs detected"
else
    python3 - << 'PYEOF' || true
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory // 1024**3}GB)")
PYEOF
fi

# =============================================================================
# 3. Core pip packages
# =============================================================================
echo ""
echo "[3/7] Installing pip packages..."

pip install --upgrade pip -q 2>&1 | tail -1
pip install numpy tqdm huggingface-hub kernels setuptools \
    "typing-extensions==4.15.0" datasets tiktoken sentencepiece -q 2>&1 | tail -1
echo "  Core packages OK"

# =============================================================================
# 4. zstandard (required for artifact sizing)
# =============================================================================
echo ""
echo "[4/7] zstandard..."
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
echo "[5/7] FlashAttention-3..."

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

    echo "  Wheels failed. Checking local flash-attention/hopper source..."
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
    :
elif python3 -c "import flash_attn; v=flash_attn.__version__; assert v.startswith('3'); print(f'  FA3 v{v} OK')" 2>/dev/null; then
    :
else
    install_fa3
fi

# =============================================================================
# 6. Dataset (sp1024)
# =============================================================================
echo ""
echo "[6/7] FineWeb dataset (sp1024)..."

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
        python3 - << PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    "sproos/parameter-golf-tokenizers",
    allow_patterns="datasets/fineweb10B_sp1024/*",
    local_dir="${WORKSPACE}/data",
)
PYEOF
    fi
    echo "  Downloaded"
fi

# =============================================================================
# 7. Cobra-specific verification
# =============================================================================
echo ""
echo "[7/7] Cobra verification..."

for f in \
    "experiments/Cobra/README.md" \
    "experiments/Cobra/cobra_harness.py" \
    "experiments/Cobra/candidates.json" \
    "experiments/Cobra/profiles/cobra_base_quality.env" \
    "experiments/Cobra/run_plan.sh"
do
    if [ ! -f "$f" ]; then
        echo "  FATAL: missing Cobra file: $f"
        exit 1
    fi
    echo "  OK: $f"
done

python3 -m py_compile experiments/Cobra/cobra_harness.py
python3 experiments/Cobra/cobra_harness.py plan >/tmp/cobra_plan_preview.txt
head -n 20 /tmp/cobra_plan_preview.txt

# =============================================================================
# Final summary
# =============================================================================
echo ""
echo "============================================"
echo " COBRA READY"
echo "============================================"
echo "Next steps:"
echo "  1) Plan only:"
echo "     bash experiments/Cobra/run_plan.sh"
echo ""
echo "  2) Dry-run one candidate command:"
echo "     python3 experiments/Cobra/cobra_harness.py run --candidate c0_base_ref --seed 1337"
echo ""
echo "  3) Execute one candidate:"
echo "     python3 experiments/Cobra/cobra_harness.py run --candidate c0_base_ref --seed 1337 --execute"
echo ""
echo "  4) Summarize Cobra logs:"
echo "     bash experiments/Cobra/summarize_logs.sh"
