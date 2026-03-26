#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/logs"

NUM_GPUS=${NUM_GPUS:-8}
SEED=${SEED:-42}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
EXPECTED_TRAIN_SHARDS=${EXPECTED_TRAIN_SHARDS:-80}

actual_train_shards=$(find ./data/datasets/fineweb10B_sp1024 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')
if [ "${actual_train_shards}" -lt "${EXPECTED_TRAIN_SHARDS}" ]; then
  echo "Preparing full FineWeb sp1024 dataset (${actual_train_shards}/${EXPECTED_TRAIN_SHARDS} train shards present)"
  python3 data/cached_challenge_fineweb.py --variant sp1024
fi
actual_train_shards=$(find ./data/datasets/fineweb10B_sp1024 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')
if [ "${actual_train_shards}" -lt "${EXPECTED_TRAIN_SHARDS}" ]; then
  echo "Expected ${EXPECTED_TRAIN_SHARDS} train shards but found ${actual_train_shards}" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/install_flash_attn_hopper.sh"

base_env() {
  env \
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    SEED="$SEED" \
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
    USE_EMA=1 EMA_FROM_INIT=1 EMA_ALPHA=0.997 XSA_LAST_N=4 \
    SMEAR_GATE=1 BIGRAM_HASH=1 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
    USE_NTK_ROPE=1 ROPE_DIMS=16 LN_SCALE=1 \
    VALUE_EMBED=1 VE_DIM=128 VE_LAYERS=9,10 \
    LATE_QAT=1 QAT_THRESHOLD=0.15 QAT_BITS=6 USE_QAT=0 \
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.04 ADAM_WD=0.04 \
    INIT_TYPE=ortho WARMDOWN_ITERS=3500 \
    TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
    EVAL_STRIDE=64 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
    COMPRESS_ALGO=zstd QUANT_BITS_MIDDLE=6 QUANT_BITS_MLP=6 GRAD_GUIDED_QUANT=0 \
    MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
    USE_FA3=1 FLASH_ATTN_BACKEND=fa3 FLASH_ATTN_STRICT=1 FLASH_ATTN_ARCH_LIST=9.0 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
    SKIP_COMPILE=0 SKIP_QUANT=0
}

run_one() {
  local slug="$1"; shift
  local run_id="higher_rank_heads_${slug}_seed${SEED}"
  base_env RUN_ID="$run_id" "$@" \
    torchrun --standalone --nproc_per_node="$NUM_GPUS" "${SCRIPT_DIR}/train_gpt.py"
  if [ -f "logs/${run_id}.txt" ]; then
    cp "logs/${run_id}.txt" "${SCRIPT_DIR}/logs/${slug}.log"
  fi
}

run_one H0_control_standard HEAD_TYPE=standard RANK_DIM=0
run_one H1_factorized_r64 HEAD_TYPE=standard RANK_DIM=64
run_one H2_factorized_r128 HEAD_TYPE=standard RANK_DIM=128
run_one H3_mos_k2_r64 HEAD_TYPE=mixture_softmax MIXTURE_SOFTMAX_K=2 MIXTURE_RANK_DIM=64 RANK_DIM=0
run_one H4_mos_k4_r64 HEAD_TYPE=mixture_softmax MIXTURE_SOFTMAX_K=4 MIXTURE_RANK_DIM=64 RANK_DIM=0
run_one H5_mos_k4_r128 HEAD_TYPE=mixture_softmax MIXTURE_SOFTMAX_K=4 MIXTURE_RANK_DIM=128 RANK_DIM=0
run_one H6_simplex_128 HEAD_TYPE=simplex SIMPLEX_DIM=128 RANK_DIM=0
