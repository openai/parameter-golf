#!/bin/bash
# =============================================================================
# Quick Smoke Test — 1xH100 (cheapest possible GPU validation)
# =============================================================================
# Run this FIRST on a 1xH100 to validate everything works before scaling to 8x.
# Cost: ~$2.50 for 5 minutes on RunPod
#
# Usage (on RunPod):
#   cd /workspace
#   git clone https://github.com/johnlennyt5/parameter-golf.git
#   cd parameter-golf
#   git checkout arch1/mamba-hybrid
#   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
#   bash run_smoke_1gpu.sh
# =============================================================================
set -euo pipefail

echo "=== 1xH100 Smoke Test for Mamba-Attention Hybrid ==="
echo ""

# Step 1: Verify CUDA kernels
echo "--- Step 1: Verify CUDA kernels ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn
print('mamba-ssm + causal-conv1d CUDA kernels: OK')
" || { echo "FAILED: Install mamba-ssm>=2.2.0 causal-conv1d>=1.4.0"; exit 1; }
echo ""

# Step 2: Run CPU tests (should take ~30s)
echo "--- Step 2: Run test suite ---"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10
echo ""

# Step 3: Quick 60-second training run on 1 GPU
echo "--- Step 3: 60-second training run (1xGPU) ---"
MAMBA_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,15,16,17 \
NUM_LAYERS=18 \
MAMBA_D_STATE=32 \
MAMBA_D_CONV=4 \
MAMBA_EXPAND=1.5 \
MAMBA_MATRIX_LR=0.015 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
WARMDOWN_ITERS=3500 \
TARGET_MB=15.9 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=60 \
RUN_ID=smoke_1gpu \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee smoke_1gpu.log

echo ""
echo "=== Smoke Test Results ==="
echo ""
grep -E "model_params|mamba_params|hybrid:|step_avg|step [0-9]+ |final_int6" smoke_1gpu.log 2>/dev/null | head -15
echo ""

# Extract step time
STEP_MS=$(grep -oP 'step_avg:\K[0-9.]+' smoke_1gpu.log | tail -1)
if [ -n "$STEP_MS" ]; then
    echo "Step time (1xGPU): ${STEP_MS}ms"
    echo "Estimated 8xGPU step time: ~$(python3 -c "print(f'{float('${STEP_MS}')/7:.1f}')") ms (rough estimate)"
    echo ""
    echo "If 8xGPU estimated step time < 100ms: PROCEED to 8xH100"
    echo "Otherwise: Review pivot strategies in MASTERPLAN"
fi
