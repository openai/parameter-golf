#!/usr/bin/env bash
# Run experiments b1, c1, c2, d1, d2 after ctrl + a1 + a2 are done (1x RTX 4090, low VRAM).
# Usage: bash commands/run_remaining_after_a2_4090.sh /path/to/train_gpt.py
set -euo pipefail

TRAIN_SCRIPT="${1:-records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py}"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "ERROR: train script not found: ${TRAIN_SCRIPT}"
  exit 1
fi

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-196608}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
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

run_one "b1_block256_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=256 BIGRAM_DIM=112

run_one "c1_bigram96_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=96 TARGET_MB=15.85

run_one "c2_bigram128_seed314" env \
  SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=128 TARGET_MB=15.90

run_one "d1_warm3500_seed314" env \
  SEED=314 WARMDOWN_ITERS=3500 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

run_one "d2_warm4500_seed314" env \
  SEED=314 WARMDOWN_ITERS=4500 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112

echo "All five runs done."
python3 commands/summarize_logs.py logs
