#!/usr/bin/env bash
# 8xH100 RunPod execution script for v62 Phase 1-C: Pentanary -> Ternary on MLP-up.
# Usage: bash run.sh <phase> <seed> <ternary_mode>
#   phase: train | eval | both
#   seed:  1337 | 1338 | 1339
#   ternary_mode: full (all 11 layers ternary) | pent (baseline)

set -euo pipefail

PHASE="${1:-both}"
SEED="${2:-1337}"
MODE="${3:-full}"

SCRIPT=records/track_10min_16mb/2026-04-09_v62_phase1c_ternary/train_gpt.py
RUN_NAME="v62_p1c_${MODE}_s${SEED}"
LOGDIR="logs/v62_p1c_${MODE}_s${SEED}"
mkdir -p "$LOGDIR"

if [[ "$MODE" == "full" ]]; then
  MLP_TYPE="ternary"
elif [[ "$MODE" == "pent" ]]; then
  MLP_TYPE="pent"
else
  echo "unknown ternary_mode: $MODE" >&2; exit 1
fi

TRAIN_ENV=(
  SEED="${SEED}" BF16_WEIGHT=0
  MATRIX_LR=0.025 TIED_EMBED_LR=0.035 SCALAR_LR=0.025
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
  MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
  TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
  ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500
  LZMA9_AFTER_RANS=1
  MLP_UP_TYPE="${MLP_TYPE}"  # Phase 1-C: ternary or pent
)

if [[ "$PHASE" == "train" || "$PHASE" == "both" ]]; then
  echo "=== [v62 Phase 1-C ${MODE}] training seed=${SEED} (MLP_UP_TYPE=${MLP_TYPE}) ==="
  env "${TRAIN_ENV[@]}" \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    --train --v61 --h100 --ema 0.997 --ema-type ema --swa \
    --seed "${SEED}" --run-name "${RUN_NAME}" \
    --log-every 200 --val-every 0 --save-every 0 \
    --data-dir data/datasets/fineweb10B_sp1024 \
    --tokenizer data/tokenizers/fineweb_1024_bpe.model \
    2>&1 | tee "${LOGDIR}/train.log"
fi

if [[ "$PHASE" == "eval" || "$PHASE" == "both" ]]; then
  CKPT="runs/${RUN_NAME}/model.rans.ptz"
  [[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
  echo "=== [v62 Phase 1-C ${MODE}] evaluating ${CKPT} ==="
  MLP_UP_TYPE="${MLP_TYPE}" python "$SCRIPT" --eval --checkpoint "$CKPT" \
    --stride 64 --batch-seqs 32 --seq-len 1024 --compile \
    --slot-lr 0.1 --slot-steps 100 --slot-lr-min 0.001 \
    --data-dir data/datasets/fineweb10B_sp1024 \
    --tokenizer data/tokenizers/fineweb_1024_bpe.model \
    2>&1 | tee "${LOGDIR}/eval.log"
  echo "=== eval done ==="
  grep -E "val_bpb|Sliding Window" "${LOGDIR}/eval.log" | tail -5
fi
