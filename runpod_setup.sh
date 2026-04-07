#!/bin/bash
# =============================================================================
# RunPod Setup Script for Parameter Golf SOTA Reproduction
# Uses YOUR fork with profiling + MLflow integrated
#
# Usage on RunPod (8xH100 SXM):
#   1. SSH into the pod
#   2. git clone https://github.com/<YOUR_USER>/parameter-golf.git /workspace/parameter_golf
#   3. cd /workspace/parameter_golf && git checkout exp/reproduce-sota
#   4. bash runpod_setup.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo " Parameter Golf -- RunPod Setup"
echo " SOTA Reproduction with Profiling + MLflow"
echo "============================================"

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR"

# ---------------------------------------------------------
# 1. System deps
# ---------------------------------------------------------
echo ""
echo "[1/4] Installing Python dependencies..."

pip install --upgrade pip -q
pip install sentencepiece zstandard numpy huggingface-hub -q

# Flash Attention 3 (Hopper)
if python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')" 2>/dev/null; then
    echo "  Flash Attention 3 already installed."
else
    echo "  Installing Flash Attention 3..."
    pip install --no-cache-dir \
        "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" \
        2>/dev/null || \
    pip install --break-system-packages flash_attn_3 \
        --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
fi

# MLflow + TensorBoard (for our tracking + profiling)
pip install mlflow tensorboard torch-tb-profiler -q

# ---------------------------------------------------------
# 2. Download dataset (sp1024, all shards)
# ---------------------------------------------------------
echo ""
echo "[2/4] Downloading FineWeb sp1024 dataset..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# ---------------------------------------------------------
# 3. Verify environment
# ---------------------------------------------------------
echo ""
echo "[3/4] Verifying environment..."
python3 -c "
import sys, torch, glob
print(f'Python:        {sys.version.split()[0]}')
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.cuda.is_available()}')
print(f'GPUs:          {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_mem // 1024**3}GB)')
try:
    from flash_attn_interface import flash_attn_func
    print(f'FlashAttn3:    OK')
except ImportError:
    print(f'FlashAttn3:    MISSING !!!')
    sys.exit(1)
import sentencepiece; print(f'SentencePiece: OK')
import zstandard;     print(f'Zstandard:     OK')
try:
    from profiling import TrainingProfiler; print(f'Profiling:     OK')
except: print(f'Profiling:     MISSING')
try:
    from tracking import ParameterGolfTracker; print(f'MLflow Track:  OK')
except: print(f'MLflow Track:  MISSING')
train = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
val   = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
print(f'Train shards:  {len(train)}')
print(f'Val shards:    {len(val)}')
if len(train) < 10: print('WARNING: Expected 10 train shards!'); sys.exit(1)
if len(val) < 1:    print('WARNING: No val shards!'); sys.exit(1)
print()
print('All checks passed.')
"

# ---------------------------------------------------------
# 4. Print run commands
# ---------------------------------------------------------
echo ""
echo "[4/4] Setup complete."
echo ""
echo "============================================"
echo " Run commands (copy-paste into terminal)"
echo "============================================"
echo ""
echo "--- Seed 314 (with profiling + GPU logging) ---"
echo ""
echo 'PROFILE=1 GPU_LOG_EVERY=100 CUDA_TIMING=1 \'
echo 'BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 TARGET_MB=15.9 \'
echo 'SEED=314 RUN_ID=R01_sota_seed314 \'
echo '  torchrun --standalone --nproc_per_node=8 train_gpt_sota.py 2>&1 | tee logs/R01_seed314.log'
echo ""
echo "--- Seed 42 ---"
echo ""
echo 'PROFILE=1 GPU_LOG_EVERY=100 CUDA_TIMING=1 \'
echo 'BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 TARGET_MB=15.9 \'
echo 'SEED=42 RUN_ID=R01_sota_seed42 \'
echo '  torchrun --standalone --nproc_per_node=8 train_gpt_sota.py 2>&1 | tee logs/R01_seed42.log'
echo ""
echo "--- Seed 999 ---"
echo ""
echo 'PROFILE=1 GPU_LOG_EVERY=100 CUDA_TIMING=1 \'
echo 'BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 TARGET_MB=15.9 \'
echo 'SEED=999 RUN_ID=R01_sota_seed999 \'
echo '  torchrun --standalone --nproc_per_node=8 train_gpt_sota.py 2>&1 | tee logs/R01_seed999.log'
echo ""
echo "--- View flame charts (after run completes) ---"
echo "tensorboard --logdir=./logs/profiler --port=6006 --bind_all"
echo ""
