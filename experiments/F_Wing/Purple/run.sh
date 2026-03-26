#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# F-Wing PURPLE — U-Net control (USE_CRAWLER=0), full X-WING N-gram stack
#
# Purpose: Clean A/B against Red. Same script, same N-gram config, same dims.
# Only difference: USE_CRAWLER=0 → standard 11L/512d U-Net (current SOTA base).
#
# Expected: ~0.4820 BPB (matches X-WING 3D Cubric record)
# If Red beats Purple: Frugendorff compression is helping. Submit Red.
# If Red matches Purple: Same BPB, smaller artifact. Still a win (headroom).
# If Red loses: Architecture penalty > compression gain. Stick with U-Net.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

SEED="${SEED:-1337}"
NPROC="${NPROC:-8}"
RESULTS_DIR="experiments/F_Wing/Purple/results"
mkdir -p "$RESULTS_DIR" checkpoints

RUN_ID="fwing_purple_$(date +%Y%m%d_%H%M%S)"
echo "================================================================"
echo "  F-Wing PURPLE — U-Net control (USE_CRAWLER=0, X-WING N-gram)"
echo "  11L/512d  seed=$SEED  600s"
echo "  RUN_ID=$RUN_ID"
echo "================================================================"

env \
  SEED="$SEED" \
  RUN_ID="$RUN_ID" \
  MAX_WALLCLOCK_SECONDS=600 \
  \
  USE_CRAWLER=0 \
  NUM_LAYERS=11 \
  \
  MODEL_DIM=512 \
  NUM_HEADS=8 \
  NUM_KV_HEADS=4 \
  MLP_MULT=3.0 \
  MLP_ACT=relu_sq \
  MLP_LEAKY_SLOPE=0.5 \
  XSA_LAST_N=4 \
  ROPE_DIMS=24 \
  LN_SCALE=1 \
  VE_ENABLED=1 \
  VE_DIM=128 \
  VE_LAYERS=9,10 \
  BIGRAM_VOCAB_SIZE=1536 \
  BIGRAM_DIM=128 \
  \
  TRAIN_SEQ_LEN=2048 \
  EVAL_SEQ_LEN=2048 \
  TRAIN_BATCH_TOKENS=786432 \
  ITERATIONS=20000 \
  WARMUP_STEPS=20 \
  WARMDOWN_ITERS=3500 \
  GRAD_CLIP_NORM=0.3 \
  MATRIX_LR=0.025 \
  SCALAR_LR=0.025 \
  TIED_EMBED_LR=0.035 \
  TIED_EMBED_INIT_STD=0.005 \
  MUON_MOMENTUM=0.99 \
  MUON_BACKEND_STEPS=5 \
  MUON_WD=0.04 \
  ADAM_WD=0.04 \
  MUON_BETA2=0.95 \
  MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=1500 \
  \
  SWA_ENABLED=1 \
  SWA_EVERY=50 \
  QAT_ENABLED=0 \
  LATE_QAT_THRESHOLD=0.5 \
  VAL_LOSS_EVERY=4000 \
  VAL_BATCH_SIZE=524288 \
  EVAL_STRIDE=64 \
  DISTILL_ENABLED=0 \
  \
  COMPLEMENT_ALPHA=0.5 \
  \
  NGRAM_EVAL_ORDER=9 \
  NGRAM_EVAL_MIN_ORDER=2 \
  NGRAM_EVAL_ALPHA=0.30 \
  NGRAM_EVAL_ADAPTIVE=1 \
  NGRAM_EVAL_ALPHA_MIN=0.20 \
  NGRAM_EVAL_ALPHA_MAX=0.75 \
  NGRAM_EVAL_ENTROPY_CENTER=3.0 \
  NGRAM_EVAL_ENTROPY_SCALE=2.0 \
  NGRAM_EVAL_MIN_COUNT=2 \
  NGRAM_EVAL_BUCKETS=8388608 \
  CUBRIC_CADENCE=32 \
  \
  torchrun --standalone --nproc_per_node="$NPROC" experiments/F_Wing/train_gpt.py \
  2>&1 | tee "$RESULTS_DIR/${RUN_ID}.log"

cp final_model.pt "checkpoints/${RUN_ID}_final.pt" 2>/dev/null || true
cp final_model.int6.ptz "checkpoints/${RUN_ID}_final.int6.ptz" 2>/dev/null || true
echo "Purple done. Log: $RESULTS_DIR/${RUN_ID}.log"
