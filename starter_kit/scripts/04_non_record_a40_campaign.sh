#!/usr/bin/env bash
set -euo pipefail

# 10-run non-record campaign focused on improving the stock A40 baseline.
# Run from repo root: bash starter_kit/scripts/04_non_record_a40_campaign.sh

cd /workspace/parameter-golf
mkdir -p logs

BASE_ENV=(
  DATA_PATH=./data/datasets/fineweb10B_sp1024
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
  VOCAB_SIZE=1024
  VAL_LOSS_EVERY=10
  TRAIN_LOG_EVERY=5
  WARMUP_STEPS=20
  ITERATIONS=60
  MAX_WALLCLOCK_SECONDS=900
)

# Format: RUN_ID|SPACE_SEPARATED_ENV_OVERRIDES
RUNS=(
  "R01_baseline_longer|TRAIN_BATCH_TOKENS=2097152 WARMDOWN_ITERS=1200"
  "R02_warmdown_short|TRAIN_BATCH_TOKENS=2097152 WARMDOWN_ITERS=200"
  "R03_warmdown_medium|TRAIN_BATCH_TOKENS=2097152 WARMDOWN_ITERS=600"
  "R04_batch_half|TRAIN_BATCH_TOKENS=1048576 WARMDOWN_ITERS=400"
  "R05_batch_quarter|TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=300"
  "R06_qk_gain_low|TRAIN_BATCH_TOKENS=1048576 WARMDOWN_ITERS=400 QK_GAIN_INIT=1.2"
  "R07_qk_gain_high|TRAIN_BATCH_TOKENS=1048576 WARMDOWN_ITERS=400 QK_GAIN_INIT=1.8"
  "R08_lr_matrix_up|TRAIN_BATCH_TOKENS=1048576 WARMDOWN_ITERS=400 MATRIX_LR=0.05 SCALAR_LR=0.04 TIED_EMBED_LR=0.05"
  "R09_lr_matrix_down|TRAIN_BATCH_TOKENS=1048576 WARMDOWN_ITERS=400 MATRIX_LR=0.03 SCALAR_LR=0.035 TIED_EMBED_LR=0.045"
  "R10_capacity_bump|TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=500 MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2"
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

python starter_kit/scripts/rank_campaign_results.py --logs-glob "logs/R*.log"
