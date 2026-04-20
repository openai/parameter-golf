#!/bin/bash
set -euo pipefail
cd /workspace/parameter-golf
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "ERROR: Requires 8 GPUs. Got $GPU_COUNT."
    exit 1
fi
HARDWARE=$(nvidia-smi -L | head -1 | sed 's/.*NVIDIA //;s/ (UUID.*//' | tr ' ' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="/workspace/logs/shep_baseline_3000_8x${HARDWARE}_${TIMESTAMP}.log"
echo "===================="
echo "SHEPHERD BASELINE B1"
echo "===================="
echo "Hardware: 8x ${HARDWARE}"
echo "Steps:    3000"
echo "Seed:     1337"
echo "Log:      $LOG"
echo "===================="
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
SEED=1337 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192nb \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_8192nb_bpe.model \
VOCAB_SIZE=8192 \
NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
WARMDOWN_ITERS=1200 \
MAX_STEPS=3000 \
MAX_WALLCLOCK_SECONDS=99999 \
torchrun --standalone --nproc_per_node=8 \
  train_breadcrumb_recur_ema_stochdepth_stepbound.py 2>&1 | tee "$LOG"
echo ""
echo "=== FINAL RESULT ==="
grep "stopping_early\|final_int8_zlib_roundtrip_exact\|Total submission size int6" "$LOG" | tail -5
