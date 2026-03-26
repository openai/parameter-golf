#!/bin/bash
set -euo pipefail
cd /workspace/parameter-golf
export RUN_ID=core_record_candidate_evalmerge_temponly_runpod
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=0
export DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model
LOG_DIR=/workspace/parameter-golf/records/track_10min_16mb/2026-03-20_core_record_candidate_evalmerge_temponly_runpod
mkdir -p "$LOG_DIR"
nohup python3 -m torch.distributed.run --standalone --nproc_per_node=8 /workspace/parameter-golf/train_gpt.py > "$LOG_DIR/train5.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG_DIR/train5.pid"
echo "started pid=$PID log=$LOG_DIR/train5.log"
