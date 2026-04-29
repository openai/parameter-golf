#!/usr/bin/env bash
# agx_queue.sh — run the AGX reference sweep overnight on the remote AGX.
# Waits for the current AGX run (by PID on remote) then launches a SEED=1337 rerun.
#
# Usage:
#   nohup bash scripts/agx_queue.sh <REMOTE_PID> > logs/sweep/orchestrator_agx_queue.log 2>&1 &

set +e

REMOTE_PID="${1:?remote pid required}"
HOST="ghost@192.168.179.191"
KEY="$HOME/.ssh/id_ed25519_agx"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "agx queue started, waiting for remote PID ${REMOTE_PID}..."
while ssh -o ConnectTimeout=5 -i "$KEY" "$HOST" "kill -0 ${REMOTE_PID} 2>/dev/null" 2>/dev/null; do
  sleep 30
done
log "remote PID ${REMOTE_PID} gone — launching seed=1337 rerun"
sleep 20

RUN_LABEL="agx_ref_1000iter_seed1337_$(date +%Y%m%d_%H%M%S)"
log "launching ${RUN_LABEL}"

ssh -i "$KEY" "$HOST" "cd /mnt/nvme && mkdir -p logs/ref && nohup env \
  DATA_DIR=/mnt/nvme/data \
  SEED=1337 \
  ITERATIONS=1000 \
  WARMUP_STEPS=10 \
  TRAIN_BATCH_TOKENS=32768 \
  TRAIN_SEQ_LEN=1024 \
  VAL_BATCH_TOKENS=131072 \
  EVAL_SEQ_LEN=1024 \
  VAL_LOSS_EVERY=0 \
  TRAIN_LOG_EVERY=50 \
  MAX_WALLCLOCK_SECONDS=7200 \
  GPTQ_CALIBRATION_BATCHES=8 \
  GPTQ_RESERVE_SECONDS=30 \
  SLIDING_WINDOW_ENABLED=1 \
  TTT_ENABLED=0 \
  TORCHDYNAMO_DISABLE=1 \
  TORCH_COMPILE_DISABLE=1 \
  PYTHONUNBUFFERED=1 \
  RUN_ID=${RUN_LABEL} \
  timeout --kill-after=30 14400 python3 -u train_gpt_agx.py > logs/ref/${RUN_LABEL}.log 2>&1 & \
  echo \"launched pid \$!\""

log "agx queue done launching"
