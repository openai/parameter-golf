#!/bin/bash
# ============================================================
# exp107: SAM inner loop for meta-TTT (replaces MetaSGD from exp106)
# Branched from exp106_metasgd-crosschunk-delta_from_exp101.
#
# Motivation: exp106's MetaSGD (66 learned per-layer inner-loop LR scales)
# converged to uniform 1.0 — no per-layer differentiation learned — while
# costing +8.6 GB peak memory and -334 training steps. SAM replaces MetaSGD
# with a DIFFERENT approach to improving the inner-loop: instead of learning
# per-layer step sizes, SAM changes the gradient DIRECTION to point toward
# flatter minima.
#
# SAM inner loop (D):
#   1. Compute gradient g at current banks (vanilla forward+backward)
#   2. Perturb banks in ascent direction: banks_pert = banks + rho * g / ||g||
#   3. Compute gradient g_sam at the perturbed point (second forward+backward)
#   4. Use g_sam for adaptation: banks' = banks - lr * g_sam
#   This finds adapted banks in flatter regions of the loss landscape.
#   If the TTT ceiling (~0.023 bpb) is set by local curvature, SAM may break it.
#
# Changes from exp106:
#   REMOVED: META_SGD_ENABLED, META_SGD_LR (66 learned params, converged to 1.0)
#   ADDED:   META_TTT_SAM_ENABLED=1, META_TTT_SAM_RHO=0.05
#   KEPT:    META_TTT_SPLIT=batch (A), META_TTT_DELTA_WEIGHT=0.3 (B)
#
# Expected memory: ~25 GB peak (vs exp106's 31.7 GB) — net win from dropping MetaSGD.
# Expected steps:  ~6800 in 4800s (vs exp106's 6686) — faster per-step from lower memory.
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="exp107_sam-inner-metattt_from_exp106"
cd /workspace/parameter-golf

# --- 8xH100 simulation ---
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-4800}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export ITERATIONS="${ITERATIONS:-7500}"
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

# --- Bigram layout (pos-conditional split, no trigram — matches exp105a baseline) ---
export POS_CONDITIONAL_BIGRAM=1
export TRIGRAM=0

# --- Wider Value Embeddings (layers 7-10) ---
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="7,8,9,10"

# --- Earlier Late QAT (threshold 0.25) ---
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

# --- EMA ---
export EMA_ENABLED=1
export EMA_DECAY=0.998
export EMA_UPDATE_EVERY=10

# --- SWA ---
export SWA_ENABLED=1
export SWA_EVERY=50

# --- Fixed momentum 0.99 (meta-TTT needs stable high momentum) ---
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

# --- Meta-TTT (FOMAML + cross-chunk (A) + delta-loss (B) + SAM inner (D)) ---
export META_TTT_ENABLED=1
export META_TTT_INNER_LR=0.002
export META_TTT_EVERY=4
export META_TTT_LOSS_WEIGHT=0.5
export META_TTT_FREEZE_BLOCKS=2
# (A) Cross-chunk split (from exp106)
export META_TTT_SPLIT=batch
# (B) Delta-loss (from exp106)
export META_TTT_DELTA_WEIGHT=0.3
# (D) SAM inner loop (exp107 — replaces MetaSGD)
export META_TTT_SAM_ENABLED=1
export META_TTT_SAM_RHO=0.05
export META_TTT_SAM_ADAPTIVE=0

# --- TTT (eval time) — SGD + cosine, unchanged ---
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
echo "=== exp107: cross-chunk (A) + delta-loss (B) + SAM inner (D, rho=${META_TTT_SAM_RHO}) ==="
python3 "${SCRIPT_DIR}/train_gpt.py" 2>&1 | tee "${SCRIPT_DIR}/logs_seed${SEED}.txt"
echo "=== ${EXP_NAME} COMPLETE ==="

# Save model artifacts for submission
echo "=== Saving model artifacts ==="
if [ -f "final_model.pt" ]; then
    cp final_model.pt "${SCRIPT_DIR}/"
    echo "Saved final_model.pt"
fi
if [ -f "final_model.int6.ptz" ]; then
    cp final_model.int6.ptz "${SCRIPT_DIR}/"
    echo "Saved final_model.int6.ptz"
fi
echo "=== Done ==="
