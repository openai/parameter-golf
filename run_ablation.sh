#!/bin/bash
# Run a single ablation experiment and record the result.
# Usage: ./run_ablation.sh <run_id> <techniques_csv> <base_checkpoint> [extra_env_vars...]
set -euo pipefail

RUN_ID="${1:?Usage: run_ablation.sh <run_id> <techniques_csv> <base_checkpoint> [env...]}"
TECHNIQUES="${2:?}"
BASE_CHECKPOINT="${3:?}"
shift 3

for var in "$@"; do export "$var"; done

export RUN_ID
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-120}"

echo "[ABLATION] Starting: $RUN_ID"
echo "[ABLATION] Techniques: $TECHNIQUES"
echo "[ABLATION] Base: $BASE_CHECKPOINT"
echo "[ABLATION] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

START_TIME=$(date +%s)

torchrun --standalone --nproc_per_node="${NPROC:-1}" train_gpt.py 2>&1 | tee "ablation_results/${RUN_ID}.log"

END_TIME=$(date +%s)
WALL=$((END_TIME - START_TIME))

VAL_LOSS=$(grep "val_loss" "ablation_results/${RUN_ID}.log" | tail -1 | grep -oP 'val_loss[= ]+\K[0-9.]+')
VAL_BPB=$(grep "val_bpb" "ablation_results/${RUN_ID}.log" | tail -1 | grep -oP 'val_bpb[= ]+\K[0-9.]+')
ARTIFACT_SIZE=$(grep -i "artifact\|compressed\|final.*size\|zlib\|zstd" "ablation_results/${RUN_ID}.log" | tail -1 | grep -oP '[0-9]+(?= bytes)' || echo "0")
STEPS=$(grep -oP 'step[= ]+\K[0-9]+' "ablation_results/${RUN_ID}.log" | tail -1 || echo "0")
GPU_CONFIG="${NPROC:-1}x$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"

python3 -c "
from ablation import AblationResult, save_result
r = AblationResult(
    run_id='$RUN_ID',
    techniques='$TECHNIQUES'.split(','),
    base_checkpoint='$BASE_CHECKPOINT',
    seed=${SEED:-1337},
    val_loss=${VAL_LOSS:-0.0},
    val_bpb=${VAL_BPB:-0.0},
    artifact_size_bytes=${ARTIFACT_SIZE:-0},
    training_steps=${STEPS:-0},
    wall_clock_seconds=$WALL,
    gpu_config='$GPU_CONFIG',
    notes='${NOTES:-}'
)
save_result(r)
"

echo "[ABLATION] Done: $RUN_ID — BPB=$VAL_BPB, Loss=$VAL_LOSS, ${WALL}s"
python3 ablation.py leaderboard
