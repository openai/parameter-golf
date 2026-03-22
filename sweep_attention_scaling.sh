#!/usr/bin/env bash
set -uo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs

RESULTS_FILE="$REPO/sweep_attention_scaling_results.txt"

COMMON_ENV=(
  PYTHONUNBUFFERED=1
  ARCH=attention
  BITLINEAR_EVAL_MODE=float
  RESET_ON_BOS=1
  D_STATE=16
  D_CONV=4
  CHUNK_SIZE=64
  EXPAND=2
  VOCAB_SIZE=1024
  TRAIN_SEQ_LEN=256
  TRAIN_BATCH_TOKENS=16384
  WARMUP_STEPS=1
  TRAIN_LOG_EVERY=50
  WARMDOWN_ITERS=50
  MAX_WALLCLOCK_SECONDS=0
  CGGR_WARMUP=0
  CGGR_RATIO=1.0
  MATRIX_LR=0.025
  SCALAR_LR=0.04
  TIED_EMBED_LR=0.05
  SEED=1337
  ATTN_LINEAR_IMPL=casted
  SSM_LINEAR_IMPL=casted
  ATTN_FFN_EXPAND=0.0
)

wait_for_existing_jobs() {
  while pgrep -af "python3 .*train_gpt.py" >/dev/null; do
    echo "Existing train_gpt.py job detected; waiting 30s..."
    pgrep -af "python3 .*train_gpt.py" || true
    sleep 30
  done
}

extract_metric() {
  local pattern="$1"
  local regex="$2"
  local log_file="$3"
  grep "$pattern" "$log_file" | tail -1 | grep -oP "$regex" || echo "N/A"
}

run_experiment() {
  local phase="$1"
  local tag="$2"
  shift 2
  local -a extra_env=("$@")
  local run_id="${phase}_${tag}_$(date +%H%M%S)"
  local log_file="logs/${run_id}.txt"
  wait_for_existing_jobs
  if env "${COMMON_ENV[@]}" "${extra_env[@]}" RUN_ID="$run_id" python3 -u train_gpt.py 2>&1 | tee "$log_file"; then
    local pre_bpb
    local post_bpb
    local mixed_bytes
    pre_bpb=$(extract_metric "val_loss:" 'val_bpb:\K[0-9.]+' "$log_file")
    post_bpb=$(extract_metric "final_mixed_zlib_roundtrip_exact" 'val_bpb:\K[0-9.]+' "$log_file")
    mixed_bytes=$(extract_metric "Total submission size mixed+zlib:" 'mixed\+zlib:\s*\K[0-9]+' "$log_file")
    echo "$run_id | $phase | $tag | OK | $pre_bpb | $post_bpb | $mixed_bytes" >> "$RESULTS_FILE"
  else
    echo "$run_id | $phase | $tag | FAILED | N/A | N/A | N/A" >> "$RESULTS_FILE"
  fi
}

{
  echo "# Attention scaling sweep - $(date)"
  echo "# run_id | phase | tag | status | pre_export_val_bpb | post_export_val_bpb | mixed_bytes"
} > "$RESULTS_FILE"

run_experiment "screen" "baseline_8x384_h8_kv4" \
  "NUM_LAYERS=8" \
  "MODEL_DIM=384" \
  "NHEADS=12" \
  "ATTN_NHEADS=8" \
  "ATTN_KV_HEADS=4" \
  "ITERATIONS=120" \
  "VAL_LOSS_EVERY=120" \
  "VAL_MAX_TOKENS=1048576"

run_experiment "screen" "deeper_10x384_h8_kv4" \
  "NUM_LAYERS=10" \
  "MODEL_DIM=384" \
  "NHEADS=12" \
  "ATTN_NHEADS=8" \
  "ATTN_KV_HEADS=4" \
  "ITERATIONS=120" \
  "VAL_LOSS_EVERY=120" \
  "VAL_MAX_TOKENS=1048576"

run_experiment "screen" "wider_8x448_h8_kv4" \
  "NUM_LAYERS=8" \
  "MODEL_DIM=448" \
  "NHEADS=14" \
  "ATTN_NHEADS=8" \
  "ATTN_KV_HEADS=4" \
  "ITERATIONS=120" \
  "VAL_LOSS_EVERY=120" \
  "VAL_MAX_TOKENS=1048576"

run_experiment "screen" "wider_10x448_h8_kv4" \
  "NUM_LAYERS=10" \
  "MODEL_DIM=448" \
  "NHEADS=14" \
  "ATTN_NHEADS=8" \
  "ATTN_KV_HEADS=4" \
  "ITERATIONS=120" \
  "VAL_LOSS_EVERY=120" \
  "VAL_MAX_TOKENS=1048576"

run_experiment "screen" "baseline_8x384_h12_kv4" \
  "NUM_LAYERS=8" \
  "MODEL_DIM=384" \
  "NHEADS=12" \
  "ATTN_NHEADS=12" \
  "ATTN_KV_HEADS=4" \
  "ITERATIONS=120" \
  "VAL_LOSS_EVERY=120" \
  "VAL_MAX_TOKENS=1048576"

run_experiment "screen" "baseline_8x384_h12_kv6" \
  "NUM_LAYERS=8" \
  "MODEL_DIM=384" \
  "NHEADS=12" \
  "ATTN_NHEADS=12" \
  "ATTN_KV_HEADS=6" \
  "ITERATIONS=120" \
  "VAL_LOSS_EVERY=120" \
  "VAL_MAX_TOKENS=1048576"

run_experiment "confirm" "wider_8x448_h8_kv4" \
  "NUM_LAYERS=8" \
  "MODEL_DIM=448" \
  "NHEADS=14" \
  "ATTN_NHEADS=8" \
  "ATTN_KV_HEADS=4" \
  "ITERATIONS=300" \
  "VAL_LOSS_EVERY=150" \
  "VAL_MAX_TOKENS=2097152"

run_experiment "confirm" "baseline_8x384_h12_kv6" \
  "NUM_LAYERS=8" \
  "MODEL_DIM=384" \
  "NHEADS=12" \
  "ATTN_NHEADS=12" \
  "ATTN_KV_HEADS=6" \
  "ITERATIONS=300" \
  "VAL_LOSS_EVERY=150" \
  "VAL_MAX_TOKENS=2097152"
