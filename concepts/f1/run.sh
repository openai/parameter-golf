#!/bin/bash
set -euo pipefail

# F1 CONCEPT BASELINE — cloned from PR #587 (submission/xsa11-clean)
# Expected (reported): seed 1337 pre-TTT 1.1203, TTT 1.1204
#
# Source commit:
#   303192e9ac65fa1673de647b02d1bb7365c37198

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
echo "============================================"
echo "  F1 BASELINE: PR #587 XSA-11 + GPTQ b64/pd002"
echo "  Seed: $SEED"
echo "  NPROC_PER_NODE: $NPROC_PER_NODE"
echo "============================================"

SEED="$SEED" \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/f1_submit_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "============================================"
echo "  DONE — check artifact size + BPB above"
echo "============================================"
ls -lh final_model.int6.ptz
