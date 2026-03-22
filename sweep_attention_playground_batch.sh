#!/usr/bin/env bash
set -uo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs

RESULTS_FILE="$REPO/sweep_attention_playground_batch_results.txt"
BEST256_CKPT="/mnt/c/Users/wrc02/Desktop/Projects/NanoGPT-Challenge/repo/artifacts/next4_14x640_h8_kv4_111139.final_model.pt"

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
  WARMUP_STEPS=0
  TRAIN_LOG_EVERY=50
  WARMDOWN_ITERS=50
  MAX_WALLCLOCK_SECONDS=0
  CGGR_WARMUP=0
  CGGR_RATIO=1.0
  MATRIX_LR=0.025
  SCALAR_LR=0.04
  TIED_EMBED_LR=0.05
  ATTN_LINEAR_IMPL=casted
  SSM_LINEAR_IMPL=casted
  ATTN_FFN_EXPAND=0.0
  ATTN_NHEADS=8
  ATTN_KV_HEADS=4
  VAL_LOSS_EVERY=110
  VAL_MAX_TOKENS=2097152
  ITERATIONS=220
  COMPILE_MODEL=0
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
  local tag="$1"
  shift
  local -a extra_env=("$@")
  local run_id="play_${tag}_$(date +%H%M%S)"
  local log_file="logs/${run_id}.txt"
  wait_for_existing_jobs
  if env "${COMMON_ENV[@]}" "${extra_env[@]}" RUN_ID="$run_id" python3 -u train_gpt.py 2>&1 | tee "$log_file"; then
    local pre_bpb post_bpb mixed_bytes
    pre_bpb=$(extract_metric "val_loss:" 'val_bpb:\K[0-9.]+' "$log_file")
    post_bpb=$(extract_metric "final_mixed_zlib_roundtrip_exact" 'val_bpb:\K[0-9.]+' "$log_file")
    mixed_bytes=$(extract_metric "Total submission size mixed+zlib:" 'mixed\+zlib:\s*\K[0-9]+' "$log_file")
    echo "$run_id | $tag | OK | $pre_bpb | $post_bpb | $mixed_bytes" >> "$RESULTS_FILE"
  else
    echo "$run_id | $tag | FAILED | N/A | N/A | N/A" >> "$RESULTS_FILE"
  fi
}

{
  echo "# Attention playground batch - $(date)"
  echo "# run_id | tag | status | pre_export_val_bpb | post_export_val_bpb | mixed_bytes"
} > "$RESULTS_FILE"

run_experiment "cont512_lr010" \
  "NUM_LAYERS=14" "MODEL_DIM=640" "NHEADS=20" "SEED=1337" \
  "TRAIN_SEQ_LEN=512" "INIT_CKPT=$BEST256_CKPT" "MATRIX_LR=0.010" "ITERATIONS=180" "VAL_LOSS_EVERY=90"

run_experiment "cont512_lr005" \
  "NUM_LAYERS=14" "MODEL_DIM=640" "NHEADS=20" "SEED=1337" \
  "TRAIN_SEQ_LEN=512" "INIT_CKPT=$BEST256_CKPT" "MATRIX_LR=0.005" "ITERATIONS=180" "VAL_LOSS_EVERY=90"

run_experiment "tied16p8_w896_seq256" \
  "NUM_LAYERS=16" "MODEL_DIM=896" "NHEADS=28" "SEED=1337" "ATTN_TIED_LAYERS=8"

run_experiment "tied12p6_w1024_seq256" \
  "NUM_LAYERS=12" "MODEL_DIM=1024" "NHEADS=32" "SEED=1337" "ATTN_TIED_LAYERS=6"

run_experiment "tied16p8_w896_seq512" \
  "NUM_LAYERS=16" "MODEL_DIM=896" "NHEADS=28" "SEED=1337" "ATTN_TIED_LAYERS=8" "TRAIN_SEQ_LEN=512"

run_experiment "mem4_14x640_seq256" \
  "NUM_LAYERS=14" "MODEL_DIM=640" "NHEADS=20" "SEED=1337" "MEMORY_TOKENS=4"

run_experiment "mem8_14x640_seq256" \
  "NUM_LAYERS=14" "MODEL_DIM=640" "NHEADS=20" "SEED=1337" "MEMORY_TOKENS=8"

run_experiment "mem8_12x768_seq512" \
  "NUM_LAYERS=12" "MODEL_DIM=768" "NHEADS=24" "SEED=1337" "MEMORY_TOKENS=8" "TRAIN_SEQ_LEN=512"

run_experiment "ema999_14x640_seq256" \
  "NUM_LAYERS=14" "MODEL_DIM=640" "NHEADS=20" "SEED=1337" "EMA_DECAY=0.999"
