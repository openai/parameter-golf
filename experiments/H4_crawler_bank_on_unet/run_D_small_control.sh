#!/bin/bash
# H4-D: Small model control (no crawler bank) — local GPU
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "flash-attention/hopper" ]; then
    export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
fi
RUN_ID="${RUN_ID:-H4D_small_ctrl_$(date +%Y%m%d_%H%M%S)}"
mkdir -p experiments/H4_crawler_bank_on_unet/results/${RUN_ID}
echo "H4-D: small control dim=256 6L | RUN_ID=$RUN_ID"
env RUN_ID="$RUN_ID" SEED=1337 \
  NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64 XSA_LAST_N=2 VE_ENABLED=1 VE_DIM=64 VE_LAYERS="4,5" \
  ROPE_DIMS=16 \
  TRAIN_BATCH_TOKENS=98304 TRAIN_SEQ_LEN=2048 \
  MAX_WALLCLOCK_SECONDS=120 WARMDOWN_ITERS=500 WARMUP_STEPS=10 \
  VAL_LOSS_EVERY=100 VAL_BATCH_SIZE=98304 \
  LATE_QAT_THRESHOLD=0.5 SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 \
  CRAWLER_BANK_ENABLED=0 \
  TTT_EVAL_ENABLED=0 \
  torchrun --standalone --nproc_per_node=1 \
    experiments/H4_crawler_bank_on_unet/GS_v7_crawler_bank_cadence.py
echo "done: $RUN_ID"
