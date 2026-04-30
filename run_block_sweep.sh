#!/usr/bin/env bash
# Block-size sweep at damp=0.01, 128-shard AR. TTT off for speed.
set -euo pipefail
cd /parameter-golf

for blk in 64 256; do
  tag="b${blk}"
  echo "=== block_size=$blk ($tag) ==="
  CKPT_PATH=wd_paired_s42.pt OUT_PTZ="requant_${tag}.int6.ptz" \
    RUN_ID="requant_${tag}" SEED=42 QK_GAIN_INIT=5.25 \
    TTT_ENABLED=0 \
    GPTQ_DAMP=0.01 GPTQ_BLOCK_SIZE=$blk GPTQ_CALIBRATION_BATCHES=64 \
    GPTQ_ALL_REDUCE=1 \
    WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 requant_eval.py 2>&1 | tee "logs/requant_${tag}.stdout" \
    | grep -E "gptq:|^quantized |quantized_sliding_window|Total submission|Traceback|RuntimeError|FAILED" \
    | sed "s|^|[$tag] |"
  echo "=== block_size=$blk done ==="
done
