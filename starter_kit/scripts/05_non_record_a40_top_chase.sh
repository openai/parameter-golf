#!/usr/bin/env bash
set -euo pipefail

# Long non-record push focused on leaderboard-chasing configs.
# Run from repo root: bash starter_kit/scripts/05_non_record_a40_top_chase.sh

cd /workspace/parameter-golf
mkdir -p logs

BASE_ENV=(
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024
  TRAIN_SEQ_LEN=1024
  VAL_LOSS_EVERY=1000
  TRAIN_LOG_EVERY=100
  WARMUP_STEPS=20
  ITERATIONS=500000
  MAX_WALLCLOCK_SECONDS=14400
  WARMDOWN_FRAC=0.2
)

# Format: RUN_ID|SPACE_SEPARATED_ENV_OVERRIDES
RUNS=(
  "A01_swiglu640_qtrbatch|MLP_ACTIVATION=swiglu MLP_HIDDEN=640 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=2 MATRIX_LR=0.05 SCALAR_LR=0.04 TIED_EMBED_LR=0.05"
  "A02_swiglu704_halfbatch|MLP_ACTIVATION=swiglu MLP_HIDDEN=704 TRAIN_BATCH_TOKENS=262144 GRAD_ACCUM_STEPS=2 MATRIX_LR=0.05 SCALAR_LR=0.04 TIED_EMBED_LR=0.05"
  "A03_swiglu640_qkgain|MLP_ACTIVATION=swiglu MLP_HIDDEN=640 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=2 QK_GAIN_INIT=1.8 MATRIX_LR=0.055 SCALAR_LR=0.04 TIED_EMBED_LR=0.05"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r run_id override_str <<< "$entry"
  read -r -a override_env <<< "$override_str"

  log_path="logs/${run_id}.log"
  echo "============================================================"
  echo "Starting ${run_id}"
  echo "Log: ${log_path}"

  env RUN_ID="$run_id" "${BASE_ENV[@]}" "${override_env[@]}" \
    torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$log_path"

  metric_line=$(grep -E "final_int8_zlib_roundtrip_exact|final_int8_zlib_roundtrip " "$log_path" | tail -1 || true)
  size_line=$(grep -E "Total submission size int8\+zlib:" "$log_path" | tail -1 || true)
  echo "Completed ${run_id}"
  echo "  ${metric_line}"
  echo "  ${size_line}"

done

python starter_kit/scripts/rank_campaign_results.py --logs-glob "logs/A*.log"
