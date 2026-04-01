#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash commands/runpod_experiments.sh /path/to/train_gpt.py
#
# Example:
#   bash commands/runpod_experiments.sh records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py
#
# This script runs a low-cost ablation matrix on 1 GPU.

TRAIN_SCRIPT="${1:-train_gpt.py}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: train script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TARGET_MB="${TARGET_MB:-15.9}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-3072}"
export XSA_LAST_N="${XSA_LAST_N:-11}"

mkdir -p logs

run_one() {
  local run_id="$1"
  shift
  echo "==== START ${run_id} ===="
  RUN_ID="${run_id}" "$@" torchrun --standalone --nproc_per_node=1 "${TRAIN_SCRIPT}" \
    2>&1 | tee "logs/${run_id}.log"
  echo "==== END ${run_id} ===="
}

# Control
run_one "ctrl_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

# A1/A2: calibration coverage
run_one "a1_calib192_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=192 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

run_one "a2_calib320_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=320 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

# B1: GPTQ block size
run_one "b1_block256_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=256 BIGRAM_DIM=112

# C1/C2: bigram dim tradeoff
run_one "c1_bigram96_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=96 TARGET_MB=15.85

run_one "c2_bigram128_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=128 TARGET_MB=15.90

# D1/D2: warmdown schedule
run_one "d1_warm3500_seed314" env \
  SEED=314 WARMDOWN_ITERS=3500 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

run_one "d2_warm4500_seed314" env \
  SEED=314 WARMDOWN_ITERS=4500 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

echo "All runs done. Logs are in logs/."
echo "Next: parse results quickly with:"
echo "  rg -n \"val_bpb|final_int8_zlib_roundtrip|artifact|compressed\" logs/*.log"
