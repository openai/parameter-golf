#!/usr/bin/env bash
set -uo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"
mkdir -p logs

RESULTS_FILE="$REPO/sweep_bos_boundary_results.txt"

COMMON_ENV=(
  PYTHONUNBUFFERED=1
  ATTN_EVERY=2
  NUM_LAYERS=12
  MODEL_DIM=512
  NHEADS=16
  D_STATE=16
  D_CONV=4
  CHUNK_SIZE=64
  EXPAND=2
  VOCAB_SIZE=1024
  TRAIN_SEQ_LEN=1024
  TRAIN_BATCH_TOKENS=65536
  WARMUP_STEPS=5
  TRAIN_LOG_EVERY=100
  WARMDOWN_ITERS=100
  MAX_WALLCLOCK_SECONDS=0
  CGGR_WARMUP=200
  CGGR_RATIO=0.5
  MATRIX_LR=0.025
  SCALAR_LR=0.04
  TIED_EMBED_LR=0.05
  SEED=1337
  TRAIN_LOADER_MODE=stream
)

SCREEN_ENV=(
  ITERATIONS=250
  VAL_LOSS_EVERY=0
  VAL_MAX_TOKENS=4194304
)

CONFIRM_ENV=(
  ITERATIONS=800
  VAL_LOSS_EVERY=400
  VAL_MAX_TOKENS=0
)

declare -A EXPERIMENT_ENVS=(
  [legacy_baseline]="ATTN_NHEADS=8 ATTN_KV_HEADS=2 ATTN_WINDOW=0 RESET_ON_BOS=0"
  [bos_baseline]="ATTN_NHEADS=8 ATTN_KV_HEADS=2 ATTN_WINDOW=0 RESET_ON_BOS=1"
  [bos_window256]="ATTN_NHEADS=8 ATTN_KV_HEADS=2 ATTN_WINDOW=256 RESET_ON_BOS=1"
  [legacy_attn16_kv4_window256]="ATTN_NHEADS=16 ATTN_KV_HEADS=4 ATTN_WINDOW=256 RESET_ON_BOS=0"
  [bos_attn16_kv4]="ATTN_NHEADS=16 ATTN_KV_HEADS=4 ATTN_WINDOW=0 RESET_ON_BOS=1"
  [bos_attn16_kv4_window256]="ATTN_NHEADS=16 ATTN_KV_HEADS=4 ATTN_WINDOW=256 RESET_ON_BOS=1"
  [bos_attn16_kv4_window256_clip9995]="ATTN_NHEADS=16 ATTN_KV_HEADS=4 ATTN_WINDOW=256 RESET_ON_BOS=1 INT4_CLIP_PERCENTILE=99.95"
)

SCREEN_ORDER=(
  legacy_baseline
  bos_baseline
  bos_window256
  legacy_attn16_kv4_window256
  bos_attn16_kv4
  bos_attn16_kv4_window256
  bos_attn16_kv4_window256_clip9995
)

{
  echo "# BOS boundary sweep results - $(date)"
  echo "# phase | run_id | tag | steps | status | pre_export_val_bpb | post_export_val_bpb | mixed_bytes | step_avg_ms"
} > "$RESULTS_FILE"

extract_metric() {
  local pattern="$1"
  local regex="$2"
  local log_file="$3"
  grep "$pattern" "$log_file" | tail -1 | grep -oP "$regex" || echo "N/A"
}

wait_for_existing_jobs() {
  while pgrep -af "python3 .*train_gpt.py" >/dev/null; do
    echo "Existing train_gpt.py job detected; waiting 30s..."
    pgrep -af "python3 .*train_gpt.py" || true
    sleep 30
  done
}

run_experiment() {
  local phase="$1"
  local tag="$2"
  local steps="$3"
  shift 3
  local -a phase_env=("$@")
  local run_id="${phase}_${tag}_$(date +%H%M%S)"
  local log_file="logs/${run_id}.txt"
  local env_string="${EXPERIMENT_ENVS[$tag]}"
  read -r -a extra_env <<< "$env_string"
  wait_for_existing_jobs

  echo ""
  echo "================================================================"
  echo "  PHASE: $phase"
  echo "  EXPERIMENT: $tag"
  echo "  run_id: $run_id"
  echo "================================================================"

  if env "${COMMON_ENV[@]}" "${phase_env[@]}" "${extra_env[@]}" RUN_ID="$run_id" python3 -u train_gpt.py 2>&1 | tee "$log_file"; then
    local pre_bpb
    local post_bpb
    local mixed_bytes
    local step_avg
    pre_bpb=$(extract_metric "step:${steps}/${steps} val_loss:" 'val_bpb:\K[0-9.]+' "$log_file")
    post_bpb=$(extract_metric "final_mixed_zlib_roundtrip_exact" 'val_bpb:\K[0-9.]+' "$log_file")
    mixed_bytes=$(extract_metric "Total submission size mixed+zlib:" 'mixed\+zlib:\s*\K[0-9]+' "$log_file")
    step_avg=$(extract_metric "step:${steps}/${steps} train_loss:" 'step_avg:\K[0-9.]+' "$log_file")
    echo "$phase | $run_id | $tag | $steps | OK | $pre_bpb | $post_bpb | $mixed_bytes | $step_avg" >> "$RESULTS_FILE"
    echo ">>> $tag complete: pre=$pre_bpb post=$post_bpb size=$mixed_bytes step_avg=${step_avg}ms"
  else
    echo "$phase | $run_id | $tag | $steps | FAILED | N/A | N/A | N/A | N/A" >> "$RESULTS_FILE"
    echo ">>> $tag failed, continuing"
  fi
}

top_screen_tags() {
  awk -F'|' '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    NR > 2 {
      phase = trim($1)
      tag = trim($3)
      status = trim($5)
      post = trim($7)
      if (phase == "screen" && status == "OK" && post != "N/A") print post "\t" tag
    }
  ' "$RESULTS_FILE" | sort -g | awk -F'\t' '!seen[$2]++ {print $2}' | head -n 3
}

for tag in "${SCREEN_ORDER[@]}"; do
  run_experiment "screen" "$tag" "250" "${SCREEN_ENV[@]}"
done

mapfile -t CONFIRM_TAGS < <(top_screen_tags)
if [ "${#CONFIRM_TAGS[@]}" -eq 0 ]; then
  CONFIRM_TAGS=(bos_baseline bos_attn16_kv4 bos_attn16_kv4_window256)
fi

echo "" | tee -a "$RESULTS_FILE"
echo "# confirm_tags: ${CONFIRM_TAGS[*]}" | tee -a "$RESULTS_FILE"

for tag in "${CONFIRM_TAGS[@]}"; do
  run_experiment "confirm" "$tag" "800" "${CONFIRM_ENV[@]}"
done

echo ""
echo "================================================================"
echo "  BOS BOUNDARY SWEEP COMPLETE"
echo "================================================================"
cat "$RESULTS_FILE"
echo ""
echo "Logs written to: $REPO/logs/"
