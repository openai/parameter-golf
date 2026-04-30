#!/usr/bin/env bash
# Damp sweep at 128-shard AR. TTT off for speed; winner gets a TTT pass after.
set -euo pipefail
cd /parameter-golf

for damp in 0.005 0.02 0.03 0.05; do
  tag="d${damp/./_}"
  echo "=== damp=$damp ($tag) ==="
  CKPT_PATH=wd_paired_s42.pt OUT_PTZ="requant_${tag}.int6.ptz" \
    RUN_ID="requant_${tag}" SEED=42 QK_GAIN_INIT=5.25 \
    TTT_ENABLED=0 \
    GPTQ_DAMP=$damp GPTQ_BLOCK_SIZE=128 GPTQ_CALIBRATION_BATCHES=64 \
    GPTQ_ALL_REDUCE=1 \
    WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 requant_eval.py 2>&1 | tee "logs/requant_${tag}.stdout" \
    | grep -E "gptq:|^quantized |quantized_sliding_window|Total submission|Traceback|RuntimeError|FAILED" \
    | sed "s|^|[$tag] |"
  echo "=== damp=$damp done ==="
done
