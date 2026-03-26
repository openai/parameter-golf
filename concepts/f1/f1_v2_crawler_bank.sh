#!/bin/bash
# F1 v2 — Accuracy profile + crawler bank at bottleneck
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
echo "============================================"
echo "  F1 v2: CRAWLER BANK + accuracy profile"
echo "  Seed: $SEED"
echo "============================================"

SEED="$SEED" \
F1_CORR_RANK=256 \
F1_CORR_SCALE_INIT=0.10 \
DISTILL_ENABLED=1 \
DISTILL_STEPS=24 \
DISTILL_LR_FACTOR=0.02 \
DISTILL_TEMPERATURE=1.5 \
DISTILL_ALPHA=0.60 \
DISTILL_KL_CLIP=10.0 \
CRAWLER_BANK_ENABLED=1 \
CRAWLER_BANK_LOOPS=2 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt_crawler_bank.py" \
    2>&1 | tee "logs/f1_v2_crawler_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

cp final_model.pt checkpoints/f1_v2_crawler_final.pt 2>/dev/null || true
cp final_model.int6.ptz checkpoints/f1_v2_crawler_final.int6.ptz 2>/dev/null || true

echo ""
echo "============================================"
echo "  DONE — F1 v2 crawler bank"
echo "============================================"
