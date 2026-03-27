#!/bin/bash
set -uo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

set -a; source /home/nesta/parameter-golf/.env; set +a

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_PATH="../../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model"
export SEED=1337
export ITERATIONS=500
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=50
export TRAIN_LOG_EVERY=10
export WARMUP_STEPS=20
export WARMDOWN_ITERS=0
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=0
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export BIGRAM_VOCAB_SIZE=1536
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=700
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export SWA_ENABLED=0
export LATE_QAT_THRESHOLD=0
export TTT_ENABLED=0
export CORE_START=3
export CORE_END=8
export CORE_QUANT_ENABLED=0
export CORE_QUANT_BITS=6
export NUM_PASSES=4
export LORA_RANK=2
export WANDB_PROJECT="parameter-golf"
export WANDB_NAME="lora_test_500step_r2_fixed"
export TORCH_COMPILE_DISABLE=1

LOG="/home/nesta/parameter-golf/lora_test_500step.log"
echo "START LoRA test: 4-pass r8, 500 steps ($(date))" | tee "$LOG"

$PYTHON train_gpt_recurrent.py \
    --feedback-mode diagonal --feedback-rank 2 \
    --residual-scale-init 0.5 \
    --jacobian-proxy-weight 0.1 \
    --no-interpass-rmsnorm \
    --lora-rank 2 \
    >> "$LOG" 2>&1

EXIT=$?
echo ""
if [ $EXIT -ne 0 ]; then
    echo "FAILED (exit=$EXIT)"
    tail -30 "$LOG"
else
    echo "=== FINAL RESULTS ==="
    grep 'val_bpb' "$LOG" | tail -5
fi

echo "FINISHED ($(date))"
