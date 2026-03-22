#!/bin/bash
# Pod startup script — run this first on any new/restarted pod
# Usage: curl -s https://raw.githubusercontent.com/mrdavtan/parameter-golf/next-gen/pod_start.sh | bash

set -e

echo "=== Pod Setup ==="

# === Environment validation (FIRST — fail fast on wrong pod) ===
echo ""
echo "=== Environment Validation ==="
ENV_OK=true

# Check Python version (need 3.12+, 3.11 has DDP optimizer issues)
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 12 ]; then
    echo "  [OK] Python $PY_VER"
else
    echo "  [FAIL] Python $PY_VER — need 3.12+ (3.11 has torch.compile + DDP incompatibility)"
    ENV_OK=false
fi

# Check PyTorch version
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
echo "  [INFO] PyTorch $TORCH_VER"
if echo "$TORCH_VER" | grep -q "2.9"; then
    echo "  [OK] PyTorch 2.9.x"
else
    echo "  [WARN] Expected PyTorch 2.9.x — got $TORCH_VER. May have compatibility issues."
fi

# Check CUDA
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "NONE")
echo "  [INFO] CUDA $CUDA_VER"

# Check GPU count
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -eq 8 ]; then
    echo "  [OK] GPUs: $GPU_COUNT"
else
    echo "  [FAIL] GPUs: $GPU_COUNT — need 8"
    ENV_OK=false
fi

# Check GPU type
GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "UNKNOWN")
echo "  [INFO] GPU: $GPU_NAME"

if [ "$ENV_OK" = false ]; then
    echo ""
    echo "=== ENVIRONMENT FAILED — terminate this pod and get one with Python 3.12 + 8xH100 ==="
    exit 1
fi

echo ""
echo "=== Installing Dependencies ==="

# Install FA3 Hopper (prebuilt wheels, fast) — gives ~15-20% faster attention
echo "Installing FA3 Hopper + FA2 fallback..."
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 -q 2>/dev/null &
FA3_PID=$!
pip install flash-attn --no-cache-dir --no-build-isolation &
FA2_PID=$!
pip install zstandard huggingface_hub sentencepiece -q

# Clone repo if needed
cd /workspace
if [ ! -d "parameter-golf/.git" ]; then
    echo "Cloning repo..."
    rm -rf parameter-golf
    git clone https://github.com/mrdavtan/parameter-golf.git
fi
cd parameter-golf

# Checkout next-gen (has all run scripts + all features)
git fetch origin
git checkout next-gen
git reset --hard origin/next-gen

# Download data and tokenizer if needed
# Use /tmp for HF cache to avoid workspace quota issues
if [ ! -d "data/datasets/fineweb10B_sp1024" ] || [ -z "$(ls data/datasets/fineweb10B_sp1024/ 2>/dev/null)" ] || [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "Downloading dataset and tokenizer..."
    rm -rf /workspace/.cache/huggingface
    HF_HOME=/tmp/hf_cache python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo "Dataset and tokenizer exist."
fi

# Wait for flash-attn installs
echo "Waiting for FA3 Hopper wheels..."
wait $FA3_PID 2>/dev/null || echo "  [WARN] FA3 wheels not available — will use FA2"
echo "Waiting for FA2 to finish compiling..."
if ! wait $FA2_PID; then
    echo "  [FAIL] flash-attn install failed. Retrying..."
    pip install flash-attn --no-cache-dir --no-build-isolation
fi

# === Preflight checks ===
echo ""
echo "=== Preflight Checks ==="
PASS=true

# Check FA3 Hopper (preferred) and FA2 (fallback)
if python3 -c "from flash_attn_interface import flash_attn_func; print('  [OK] FA3 Hopper (flash_attn_interface)')" 2>/dev/null; then
    :
elif python3 -c "from flash_attn import flash_attn_func; print('  [OK] FA2 (flash_attn — FA3 not available)')" 2>/dev/null; then
    :
else
    echo "  [FAIL] neither FA3 nor FA2 installed"
    PASS=false
fi

# Check other deps
if python3 -c "import zstandard, sentencepiece, huggingface_hub; print('  [OK] zstandard, sentencepiece, huggingface_hub')" 2>/dev/null; then
    :
else
    echo "  [FAIL] missing Python deps"
    PASS=false
fi

# Check dataset (need 80 train shards + val shards)
if [ -d "data/datasets/fineweb10B_sp1024" ]; then
    TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
    VAL_SHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
    if [ "$TRAIN_SHARDS" -ge 80 ] && [ "$VAL_SHARDS" -ge 1 ]; then
        echo "  [OK] dataset ($TRAIN_SHARDS train + $VAL_SHARDS val shards)"
    else
        echo "  [FAIL] dataset incomplete ($TRAIN_SHARDS train, $VAL_SHARDS val — need 80+1)"
        PASS=false
    fi
else
    echo "  [FAIL] dataset missing"
    PASS=false
fi

# Check tokenizer
if [ -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "  [OK] tokenizer"
else
    echo "  [FAIL] tokenizer missing"
    PASS=false
fi

# Check git branch and run scripts
BRANCH=$(git branch --show-current)
COMMIT=$(git log --oneline -1)
echo "  [OK] branch: $BRANCH ($COMMIT)"

for SCRIPT in run_baseline.sh run_oneshot.sh run_moonshot.sh; do
    if [ -f "$SCRIPT" ]; then
        echo "  [OK] $SCRIPT exists"
    else
        echo "  [FAIL] $SCRIPT missing"
        PASS=false
    fi
done

# torch.compile smoke test
echo "  [..] torch.compile smoke test..."
SMOKE_OK=$(python3 -c "
import torch, torch.nn.functional as F
from torch import nn
m = nn.Linear(512, 512).cuda().bfloat16()
cm = torch.compile(m, dynamic=False)
x = torch.randn(2, 8, 512, device='cuda', dtype=torch.bfloat16)
y = cm(x)
print('ok')
" 2>/dev/null || echo "fail")
if [ "$SMOKE_OK" = "ok" ]; then
    echo "  [OK] torch.compile works"
else
    echo "  [WARN] torch.compile may have issues"
fi

if [ "$PASS" = false ]; then
    echo ""
    echo "=== PREFLIGHT FAILED — fix errors above ==="
    exit 1
fi

echo ""
echo "=== All checks passed. Ready to run. ==="
echo ""
echo "Environment: Python $PY_VER | PyTorch $TORCH_VER | CUDA $CUDA_VER | $GPU_COUNT x $GPU_NAME"
echo ""
echo "Run scripts available (each handles git checkout + exports + launch):"
echo ""
echo "  bash run_aggressive_ttt.sh 1337   # Two-phase DDP TTT — targets 1.12x"
echo "  bash run_moonshot.sh 1337         # + Reptile + VE — targets 1.11x"
echo "  bash run_baseline.sh              # Proven 1.133 config — safe fallback"
echo ""
echo "Kill if step_avg@200 > 85ms (bad pod)"
