#!/bin/bash
# ============================================================================
# Parameter Golf — #1 Leaderboard Stack (1.1147 BPB)
# Architecture: 11L dim=512 vocab=1024 sp1024
#   • XSA on all 11 layers
#   • LeakyReLU(0.5)² MLP
#   • BigramHash 3072×112
#   • Full Hessian GPTQ with AR self-generated calibration
#   • Partial RoPE (16/64 dims), LN Scale 1/√(layer+1)
#   • VE128 on layers 9-10
#   • Parameter Banking + Parallel Muon
#   • SWA every 50 steps
#   • WARMDOWN_ITERS=4000
#   • LZMA preset=9
#   • Selective ±1 pruning to TARGET_MB=15.9
#
# Seeds: overridable via SEED env var (default 42)
# ============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

# ── Data (sp1024, vocab=1024 — the competition standard) ─────────────────────
export DATA_PATH="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024

# ── Architecture (exact #1 defaults — all baked into train_gpt_winner.py) ────
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0

# XSA on all 11 layers (novel contribution of #1)
export XSA_LAST_N=11

# BigramHash: 3072×112 (#1 setting; up from 2048×128 in PR #549)
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112

# VE (Vocabulary Embedding) on layers 9-10
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Partial RoPE + LN Scale
export ROPE_DIMS=16
export LN_SCALE=1

# ── Training budget ───────────────────────────────────────────────────────────
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
export ITERATIONS=20000
export WARMUP_STEPS=20
export SEED=${SEED:-42}

# ── Batch ─────────────────────────────────────────────────────────────────────
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048

# ── Optimizer (Parameter Banking + Parallel Muon — #1 settings) ──────────────
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=5
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export WARMDOWN_ITERS=4000

# ── Weight averaging ──────────────────────────────────────────────────────────
export SWA_ENABLED=1
export SWA_EVERY=50
export LAWA_ENABLED=0

# ── GPTQ (AR self-generated calibration — no external data) ──────────────────
export GPTQ_CALIB_BATCHES=256   # 64 seqs × 4 grad_accum = 256 batches
export GPTQ_BLOCK_SIZE=128

# ── Selective pruning target ──────────────────────────────────────────────────
export TARGET_MB=15.9

# ── Eval ──────────────────────────────────────────────────────────────────────
export EVAL_STRIDE=64
export VAL_LOSS_EVERY=4000
export TRAIN_LOG_EVERY=500

# ── Run ID & logging ──────────────────────────────────────────────────────────
export RUN_ID="winner_s${SEED}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="${DIR}/logs/${RUN_ID}.log"

echo "=========================================================================="
echo "  Parameter Golf — #1 Stack Run"
echo "  RUN_ID : ${RUN_ID}"
echo "  SEED   : ${SEED}"
echo "  MODEL  : 11L dim=512 vocab=1024 XSA-all LeakyReLU² BigramHash3072×112"
echo "  BUDGET : ${MAX_WALLCLOCK_SECONDS}s"
echo "  BATCH  : ${TRAIN_BATCH_TOKENS} tokens/step  seq=${TRAIN_SEQ_LEN}"
echo "  LR     : matrix=${MATRIX_LR}  warmdown=${WARMDOWN_ITERS} steps"
echo "  GPTQ   : AR self-gen calibration  target=${TARGET_MB}MB"
echo "=========================================================================="

# ── Launch ────────────────────────────────────────────────────────────────────
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt_winner.py 2>&1 | tee "$LOG"

# Preserve per-seed artifacts (winner uses final_model.int6.ptz)
cp final_model.int6.ptz   "logs/${RUN_ID}_model.int6.ptz"  2>/dev/null || true
cp final_model.pt         "logs/${RUN_ID}_model.pt"         2>/dev/null || true

echo "=== DONE ==="
echo "Log      : $LOG"
echo "Artifact : logs/${RUN_ID}_model.int6.ptz"
