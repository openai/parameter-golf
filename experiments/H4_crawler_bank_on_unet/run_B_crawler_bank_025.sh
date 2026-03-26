#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# H4-B: GS v7 + CRAWLER BANK (1 shared block × 2 loops) — 0.25 scale
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
RUN_ID="${RUN_ID:-H4B_gsv7_crawler_bank_$(date +%Y%m%d_%H%M%S)}"

mkdir -p experiments/H4_crawler_bank_on_unet/results/${RUN_ID} checkpoints

echo "H4-B: GS v7 + crawler bank (1×2 at bottleneck) | Scale 0.25 | RUN_ID=$RUN_ID"
env \
  RUN_ID="$RUN_ID" SEED="$SEED" \
  MAX_WALLCLOCK_SECONDS=150 WARMDOWN_ITERS=875 \
  VAL_LOSS_EVERY=500 \
  CRAWLER_BANK_ENABLED=1 CRAWLER_BANK_LOOPS=2 \
  TTT_EVAL_ENABLED=0 \
  torchrun --standalone --nproc_per_node="$NPROC" \
    experiments/H4_crawler_bank_on_unet/GS_v7_crawler_bank.py

cp final_model.pt checkpoints/${RUN_ID}_final.pt 2>/dev/null || true
cp final_model.intq.ptz checkpoints/${RUN_ID}_final.intq.ptz 2>/dev/null || true
echo "done: $RUN_ID"
