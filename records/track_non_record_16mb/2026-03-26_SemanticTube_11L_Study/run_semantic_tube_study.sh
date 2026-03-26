#!/usr/bin/env bash
set -euo pipefail

# Confirmatory runner for the semantic-tube study.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/public_logs"

ROLE=${ROLE:-tube}             # control | tube
# Defaults target the strongest matched pair from the study: S2 vs S3 at seq2048.
RUN_LABEL=${RUN_LABEL:-tube_confirm}
SEED=${SEED:-42}
NUM_GPUS=${NUM_GPUS:-8}
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-2048}
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-524288}
SKIP_QUANT=${SKIP_QUANT:-0}
SKIP_COMPILE=${SKIP_COMPILE:-0}
EXPECTED_TRAIN_SHARDS=${EXPECTED_TRAIN_SHARDS:-80}

case "${ROLE}" in
  control)
    LAMBDA_TUBE=${LAMBDA_TUBE:-0.0}
    ;;
  tube)
    LAMBDA_TUBE=${LAMBDA_TUBE:-0.0005}
    ;;
  *)
    echo "Unknown ROLE=${ROLE}; use control or tube" >&2
    exit 1
    ;;
esac

RUN_ID=${RUN_ID:-semantic_tube_${RUN_LABEL}_${ROLE}_seed${SEED}}

actual_train_shards=$(find ./data/datasets/fineweb10B_sp1024 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')
if [ "${actual_train_shards}" -lt "${EXPECTED_TRAIN_SHARDS}" ]; then
  echo "Preparing full FineWeb sp1024 dataset (${actual_train_shards}/${EXPECTED_TRAIN_SHARDS} train shards present)"
  python3 data/cached_challenge_fineweb.py --variant sp1024
  actual_train_shards=$(find ./data/datasets/fineweb10B_sp1024 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')
  if [ "${actual_train_shards}" -lt "${EXPECTED_TRAIN_SHARDS}" ]; then
    echo "Expected ${EXPECTED_TRAIN_SHARDS} train shards but found ${actual_train_shards}" >&2
    exit 1
  fi
fi

bash "${SCRIPT_DIR}/install_flash_attn_hopper.sh"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" OMP_NUM_THREADS="$OMP_NUM_THREADS" RUN_ID="$RUN_ID" SEED="$SEED" MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" VAL_LOSS_EVERY="$VAL_LOSS_EVERY" TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" TRAIN_BATCH_TOKENS="$TRAIN_BATCH_TOKENS" SKIP_QUANT="$SKIP_QUANT" SKIP_COMPILE="$SKIP_COMPILE" LAMBDA_TUBE="$LAMBDA_TUBE" NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 USE_EMA=1 EMA_FROM_INIT=1 EMA_ALPHA=0.997 XSA_LAST_N=4 SMEAR_GATE=1 USE_NTK_ROPE=1 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.04 ADAM_WD=0.04 INIT_TYPE=ortho WARMDOWN_ITERS=3000 GRAD_GUIDED_QUANT=1 TRAIN_LOG_EVERY=200 LOG_TUBE_METRICS=1 USE_FA3=1 FLASH_ATTN_BACKEND=fa3 FLASH_ATTN_STRICT=1 FLASH_ATTN_ARCH_LIST=9.0 DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node="$NUM_GPUS" "${SCRIPT_DIR}/train_gpt.py"

if [ -f "logs/${RUN_ID}.txt" ]; then
  cp "logs/${RUN_ID}.txt" "${SCRIPT_DIR}/public_logs/${RUN_ID}.txt"
fi
