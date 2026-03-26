#!/bin/bash
set -euo pipefail

# Car 03: quality lane (starts from same gold baseline)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:-f1_garage_car03_quality_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="logs/${RUN_ID}.log"

echo "============================================"
echo "  F1 SOTA GARAGE :: CAR03 QUALITY LANE"
echo "  Seed: $SEED"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  RUN_ID: $RUN_ID"
echo "============================================"

SEED="$SEED" \
RUN_ID="$RUN_ID" \
F1_CORR_RANK="${F1_CORR_RANK:-0}" \
DISTILL_ENABLED="${DISTILL_ENABLED:-0}" \
MLP_ACT="${MLP_ACT:-leaky_relu_sq}" \
MLP_LEAKY_SLOPE="${MLP_LEAKY_SLOPE:-0.5}" \
XSA_LAST_N="${XSA_LAST_N:-4}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}" \
TTT_FREEZE_BLOCKS="${TTT_FREEZE_BLOCKS:-0}" \
TTT_GRAD_CLIP="${TTT_GRAD_CLIP:-0.8}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${LOG_PATH}"

echo ""
echo "============================================"
echo "  DONE — check artifact size + BPB above"
echo "============================================"
ls -lh final_model.int6.ptz
