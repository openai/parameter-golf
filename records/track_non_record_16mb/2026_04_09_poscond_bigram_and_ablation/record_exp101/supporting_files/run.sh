#!/bin/bash
# ============================================================
# exp101: position-conditional bigram + trigram (from exp95)
# Two changes vs exp95 (both target the bigram hash layout, zero param cost):
#   1. POS_CONDITIONAL_BIGRAM=1: split the 4095 usable buckets into two
#      disjoint halves keyed on has_leading_space[current_token]. ws-current
#      pairs hash into [0, 2047), non-ws-current pairs into [2047, 4094).
#      Gives word-start bigrams their own exclusive rows (no contamination
#      from the much-more-numerous within-word pairs). Should let the bigram
#      learn meaningful word-start signal that exp95 was forced to suppress
#      via word_start_boost -> 0.007.
#   2. TRIGRAM=1: enable the (t-2, t-1, t) lookup that reuses the same table
#      (zero extra params). Adds higher-order local context. When combined
#      with pos_conditional, trigram lookups also respect the split.
#   Everything else identical to exp95. word_start_boost stays in place
#   (still useful as a safety rail in case ws buckets still end up noisy).
#   Net param change: +0.
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="exp101_poscond-bigram-trigram_from_exp95"
cd /workspace/parameter-golf

# --- 8xH100 simulation ---
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-6000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export ITERATIONS="${ITERATIONS:-9000}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-2500}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"

# --- Eval ---
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=128
export SEED="${SEED:-42}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-3000}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-262144}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-500}"

# --- Architecture ---
export NUM_LAYERS=11
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1

# --- Smaller bigram (saves ~1.5 MB → eliminates ±1 pruning) ---
export BIGRAM_VOCAB_SIZE=4096
export BIGRAM_DIM=64

# --- exp101: bigram layout changes ---
# POS_CONDITIONAL_BIGRAM=1: split buckets ws/non-ws (see BigramHashEmbedding docstring)
# TRIGRAM=1: enable (t-2,t-1,t) lookup in the same table, zero extra params
export POS_CONDITIONAL_BIGRAM=1
export TRIGRAM=0

# --- Wider Value Embeddings (layers 7-10, was 9-10) ---
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="7,8,9,10"

# --- Earlier Late QAT (threshold 0.25, was 0.15) ---
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0.25

# --- Adaptive Warmdown ---
export ADAPTIVE_WARMDOWN=1
export ADAPTIVE_WARMDOWN_EMA=0.99
export ADAPTIVE_WARMDOWN_THRESHOLD=0.0005
export ADAPTIVE_WARMDOWN_MIN_STEPS=2000

# --- Learning rates ---
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035

# --- Weight decay ---
export MUON_WD=0.04
export ADAM_WD=0.04

# --- EMA (tighter focus on converged weights) ---
export EMA_ENABLED=1
export EMA_DECAY=0.998
export EMA_UPDATE_EVERY=10

# --- SWA ---
export SWA_ENABLED=1
export SWA_EVERY=50

# --- Fixed momentum 0.99 (meta-TTT needs stable high momentum) ---
# Cycling would dilute the weak FOMAML gradient signal (3x faster forgetting at 0.97)
export MOMENTUM_CYCLIC=0
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500

# --- Newton-Schulz ---
export MUON_BACKEND_STEPS=5

# --- Grad clipping ---
export GRAD_CLIP_NORM=0.3

# --- GPTQ ---
export GPTQ_CALIB_BATCHES=256
export GPTQ_BLOCK_SIZE=128
export TARGET_MB=15.9

# --- Meta-TTT (FOMAML) — EVERY=2, disabled in warmdown ---
export META_TTT_ENABLED=1
export META_TTT_INNER_LR=0.002
export META_TTT_EVERY=4
export META_TTT_LOSS_WEIGHT=0.5
export META_TTT_FREEZE_BLOCKS=2

# --- TTT (eval time) — AdamW, flat LR, larger chunks ---
export TTT_ENABLED=1
export TTT_LR=0.004
export TTT_EPOCHS=4
export TTT_CHUNK_TOKENS=65536
export TTT_FREEZE_BLOCKS=2
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=16
export TTT_GRAD_CLIP=1.0

export RUN_ID="${EXP_NAME}_seed${SEED}"
echo "=== ${EXP_NAME} seed=${SEED} ==="
echo "=== Size-opt, TTT-opt (AdamW+flat LR), Meta-TTT 2x ==="
python3 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/logs_seed${SEED}.txt"
echo "=== ${EXP_NAME} COMPLETE ==="
