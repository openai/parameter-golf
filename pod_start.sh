#!/bin/bash
# Pod startup script — run this first on any new/restarted pod
# Usage: curl -s https://raw.githubusercontent.com/mrdavtan/parameter-golf/perlayer-lr-stack/pod_start.sh | bash

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
elif [ "$GPU_COUNT" -ge 1 ]; then
    echo "  [WARN] GPUs: $GPU_COUNT (competition runs need 8, but setup will continue)"
else
    echo "  [FAIL] No GPUs detected"
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
echo "Installing FA3 Hopper..."
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 -q 2>&1 || echo "  [WARN] FA3 wheels failed — will try again after repo setup"
echo "Installing FA2 fallback (compiles from source, takes a few minutes)..."
pip install flash-attn --no-cache-dir --no-build-isolation -q 2>&1 &
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

# Checkout causal-ttt (has all run scripts + all features)
git fetch origin
git checkout perlayer-lr-stack
git reset --hard origin/perlayer-lr-stack

# Download data and tokenizer if needed
# Use /tmp for HF cache to avoid workspace quota issues
if [ ! -d "data/datasets/fineweb10B_sp1024" ] || [ -z "$(ls data/datasets/fineweb10B_sp1024/ 2>/dev/null)" ] || [ ! -f "data/tokenizers/fineweb_1024_bpe.model" ]; then
    echo "Downloading dataset and tokenizer..."
    rm -rf /workspace/.cache/huggingface
    HF_HOME=/tmp/hf_cache python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo "Dataset and tokenizer exist."
fi

# Wait for FA2 compile
echo "Waiting for FA2 to finish compiling..."
if ! wait $FA2_PID; then
    echo "  [WARN] FA2 background install failed. Retrying..."
    cd /workspace/parameter-golf && pip install flash-attn --no-cache-dir --no-build-isolation
fi

# Retry FA3 if not available (install from a known-good working directory)
cd /workspace/parameter-golf
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "Retrying FA3 Hopper install..."
    pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 2>&1 || echo "  [WARN] FA3 not available — will use FA2"
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

for SCRIPT in run_no_ttt.sh run_causal_ttt.sh; do
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
echo "Run scripts:"
echo ""
if [ "$GPU_COUNT" -ge 8 ]; then
    echo "  bash run_no_ttt.sh 1337        # 8xGPU: full stack, no TTT (competition submission)"
    echo "  bash run_neural_cache.sh 1337  # 8xGPU: same + neural cache eval (A/B test)"
    echo ""
    echo "Kill if step_avg@200 > 85ms (bad pod)"
else
    echo "  bash run_no_ttt_1gpu.sh 1337   # 1xGPU: full stack, no TTT (test run, ~1hr)"
fi
