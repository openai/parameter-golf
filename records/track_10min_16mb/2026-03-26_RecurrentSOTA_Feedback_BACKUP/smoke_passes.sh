#!/bin/bash
set -euo pipefail

PYTHON="/home/nesta/parameter-golf/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export DATA_PATH="../../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model"
export SEED=1337
export ITERATIONS=30
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=15
export TRAIN_LOG_EVERY=10
export WARMUP_STEPS=5
export WARMDOWN_ITERS=5
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
export MUON_MOMENTUM_WARMUP_STEPS=5
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export SWA_ENABLED=0
export LATE_QAT=0
export TTT_ENABLED=0
export CORE_START=3
export CORE_END=8
export CORE_QUANT_ENABLED=0

for PASSES in 2 3 4; do
    echo ""
    echo "========================================"
    echo "  NUM_PASSES=$PASSES (30 steps)"
    echo "========================================"
    export NUM_PASSES=$PASSES
    $PYTHON train_gpt_recurrent.py \
        --feedback-mode diagonal --feedback-rank 2 \
        --residual-scale-init 0.5 \
        --jacobian-proxy-weight 0.01
done

echo ""
echo "=== All pass tests complete ==="
