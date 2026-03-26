#!/bin/bash
set -euo pipefail

# F1 SOTA SPEED-SAFE TEST
# Base: PR #587 commit 303192e9ac65fa1673de647b02d1bb7365c37198 (clean copy)
# Only deltas from baseline run:
#   1) XSA_LAST_N=4
#   2) BIGRAM_VOCAB_SIZE=1536

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_ID="${RUN_ID:-f1_sota_x4_bg1536_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="logs/${RUN_ID}.log"

echo "============================================"
echo "  F1 SOTA BASE + 2 SAFE SPEED KNOBS"
echo "  Seed: $SEED"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  RUN_ID: $RUN_ID"
echo "============================================"

SEED="$SEED" \
RUN_ID="$RUN_ID" \
XSA_LAST_N="${XSA_LAST_N:-4}" \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-1536}" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "${LOG_PATH}"

echo ""
echo "============================================"
echo "  DONE — check artifact size + BPB above"
echo "============================================"
ls -lh final_model.int6.ptz
