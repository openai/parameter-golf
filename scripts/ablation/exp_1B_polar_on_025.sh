#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# EXP-1B: Blending Mechanism — Polar decomposition
# Parent: RC-0    Scale: 0.25    Mode: EXPLAIN
# Hypothesis: Polar decomposition preserves activation energy during
#   double-firing consensus. C-step loss should be lower by step 500.
# Override vs RC-0: POLAR_ENABLED=1
# ══════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    if [ -d "flash-attention/hopper" ]; then
        export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
    else
        echo "ERROR: flash_attn_interface not found." && exit 1
    fi
fi

NPROC="${NPROC:-8}"
SEED="${SEED:-1337}"
RUN_ID="${RUN_ID:-exp_1B_polar_on_025_$(date +%Y%m%d_%H%M%S)}"

mkdir -p results/autoruns/${RUN_ID} checkpoints

echo "EXP-1B: Polar blending test | Scale 0.25 | RUN_ID=$RUN_ID"
env \
  RUN_ID="$RUN_ID" SEED="$SEED" \
  NUM_FLAT_LAYERS=4 NUM_CRAWLER_LAYERS=2 CRAWLER_LOOPS=2 CRAWLER_MLP_MULT=4 \
  MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=4 VOCAB_SIZE=1024 \
  CRAWLER_CADENCE_EARLY=2 CRAWLER_CADENCE_MAIN=2 CRAWLER_CADENCE_LATE=2 \
  TRIGRAM_VOCAB_SIZE=8192 TRIGRAM_DIM=128 \
  XSA_LAST_N=2 ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=0,1 \
  TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 \
  TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
  ITERATIONS=20000 WARMUP_STEPS=20 GRAD_CLIP_NORM=0.3 \
  MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=625 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 TIED_EMBED_INIT_STD=0.005 \
  MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5 MUON_WD=0.04 ADAM_WD=0.04 MUON_BETA2=0.95 \
  MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15 \
  GPTQ_BLOCK_SIZE=128 GPTQ_PERCDAMP=0.01 GPTQ_CALIBRATION_SAMPLES=256 \
  QUANT_INT_CATEGORIES=mlp,attn QUANT_MLP_CLIP_RANGE=15 QUANT_ATTN_CLIP_RANGE=31 \
  QUANT_EMBED_CLIP_RANGE=31 QUANT_OTHER_CLIP_RANGE=31 QUANT_ARTIFACT_NAME=final_model.intq.ptz \
  TTT_BURST_ENABLED=0 DISTILL_ENABLED=0 \
  EVAL_STRIDE=64 VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=524288 \
  DIAG_FIXED_CADENCE=2 DIAG_FAST_VAL=1 \
  POLAR_ENABLED=1 \
  DIAG_CSV_PATH="results/autoruns/${RUN_ID}/diag.csv" \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt_diag_ts_polar.py

cp final_model.pt checkpoints/${RUN_ID}_final.pt 2>/dev/null || true
cp final_model.intq.ptz checkpoints/${RUN_ID}_final.intq.ptz 2>/dev/null || true
echo "done: results/autoruns/${RUN_ID}/diag.csv"
