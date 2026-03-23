#!/bin/bash
set -euo pipefail

# SUBMISSION RUN — XSA-11 + GPTQ block64/pd002
# Expected: ~1.1201 BPB, ~15.4 MB
#
# Changes vs GS baseline (1.1206 BPB, 15.56 MB):
#   - XSA_LAST_N=11 (was 4) → -0.0006 BPB
#   - GPTQ block_size=64, percdamp=0.002 → ~570KB smaller artifact
#   - Net: better BPB AND smaller artifact

cd /workspace/parameter-golf
export PYTHONPATH="/workspace/parameter-golf/flash-attention/hopper:${PYTHONPATH:-}"

python3 -c "from flash_attn_interface import flash_attn_func; import zstandard; print('deps OK')"

SEED="${SEED:-1337}"
echo "============================================"
echo "  SUBMISSION: XSA-11 + GPTQ b64/pd002"
echo "  Seed: $SEED"
echo "============================================"

SEED="$SEED" \
torchrun --standalone --nproc_per_node=8 \
    train_gpt_v7_submit.py \
    2>&1 | tee "logs/submit_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "============================================"
echo "  DONE — check artifact size + BPB above"
echo "============================================"
ls -lh final_model.int6.ptz
