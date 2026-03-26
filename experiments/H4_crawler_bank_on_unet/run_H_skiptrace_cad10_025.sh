#!/bin/bash
# H4-H: Crawler bank with SKIPTRACE — fire every 10 steps, inject decaying delta between
# Learned decay rate + learned injection scale. Near-zero overhead.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "flash-attention/hopper" ]; then
    export PYTHONPATH="$(pwd)/flash-attention/hopper:${PYTHONPATH:-}"
fi
NPROC="${NPROC:-8}"
RUN_ID="${RUN_ID:-H4H_skiptrace_$(date +%Y%m%d_%H%M%S)}"
mkdir -p experiments/H4_crawler_bank_on_unet/results/${RUN_ID} checkpoints
echo "H4-H: Skiptrace cad=10 (8L/384d) | Scale 0.25 | RUN_ID=$RUN_ID"
env RUN_ID="$RUN_ID" SEED=1337 \
  NUM_LAYERS=8 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=3 \
  BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64 XSA_LAST_N=4 \
  VE_ENABLED=1 VE_DIM=64 VE_LAYERS="6,7" \
  ROPE_DIMS=16 LN_SCALE=1 \
  TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
  MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=875 WARMUP_STEPS=10 \
  VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=524288 \
  LATE_QAT_THRESHOLD=0.5 SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 \
  CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 CRAWLER_BANK_CADENCE=10 \
  TTT_EVAL_ENABLED=0 \
  torchrun --standalone --nproc_per_node="$NPROC" \
    experiments/H4_crawler_bank_on_unet/GS_v7_crawler_bank_cadence.py
cp final_model.pt checkpoints/${RUN_ID}_final.pt 2>/dev/null || true
echo "done: $RUN_ID"
