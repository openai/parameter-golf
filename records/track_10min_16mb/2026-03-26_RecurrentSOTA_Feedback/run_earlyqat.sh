#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f /home/nesta/parameter-golf/.env ]; then
    set -a; source /home/nesta/parameter-golf/.env; set +a
fi

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- Data paths ---
export DATA_PATH="../../../data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="../../../data/tokenizers/fineweb_1024_bpe.model"

# --- Architecture (11 layers, matching baseline capacity) ---
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export BIGRAM_VOCAB_SIZE=0
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=0
export VE_DIM=128
export VE_LAYERS="9,10"

# --- Training schedule (progressive 1->4 passes, wallclock-capped at 600s on 8xH100) ---
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=50
export WARMUP_STEPS=24
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=64

# --- Optimizer (matches SOTA) ---
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# --- Weight averaging & quantization ---
# EARLY QAT: threshold 0.25 (vs 0.15 in winning config) to reduce weight entropy
export SWA_ENABLED=1
export SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.30

# --- TTT (matches SOTA, freeze_blocks=0) ---
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=49152
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# --- Recurrence (our contribution) ---
export CORE_START=4
export CORE_END=7
export NUM_PASSES=1
export EVAL_PASSES=4
export CORE_QUANT_ENABLED=0
# Progressive: 1-pass until step 4500, then ramp 2->3->4
export PASSES_SCHEDULE="0:1,4500:2,5500:3,6000:4"

export SEED=1337
export RUN_ID="earlyqat_v3"

torchrun --standalone --nproc_per_node=8 train_gpt.py \
    --feedback-mode diagonal --feedback-rank 2 \
    --residual-scale-init 0.5 \
    --jacobian-proxy-weight 0.1 \
    --no-interpass-rmsnorm \
    2>&1 | tee logs/earlyqat_v3.txt
