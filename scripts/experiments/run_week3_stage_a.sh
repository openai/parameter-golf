#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_a_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

mkdir -p "$OUT_DIR"
touch "$RUNNER_LOG"

printf "phase\trun_id\tstatus\tconfig\tmask_schedule\ttrain_timestep_sampling\tparameterization\tself_conditioning\tloss_reweighting\tcheckpoint\tbest_checkpoint\tlog_path\n" > "$SUMMARY_TSV"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  local msg="$1"
  printf "[%s] %s\n" "$(timestamp)" "$msg" | tee -a "$RUNNER_LOG"
}

run_with_config() {
  local phase="$1"
  local run_id="$2"
  local config_path="$3"
  local mask_schedule="$4"
  local timestep_sampling="$5"
  local parameterization="$6"
  local self_conditioning="$7"
  local loss_reweighting="$8"

  local train_log_path="$OUT_DIR/${run_id}_diffusion.txt"
  local last_ckpt="$OUT_DIR/${run_id}_diffusion_last_mlx.npz"
  local best_ckpt="$OUT_DIR/${run_id}_diffusion_best_mlx.npz"

  log "Starting $phase run_id=$run_id config=$(basename "$config_path") schedule=$mask_schedule timestep_sampling=$timestep_sampling"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN: skipping train_diffusion.py for $run_id"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$phase" "$run_id" "dry-run" "$(basename "$config_path")" "$mask_schedule" "$timestep_sampling" \
      "$parameterization" "$self_conditioning" "$loss_reweighting" "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
    return 0
  fi

  if (
    set -a
    source "$config_path"
    set +a
    export RUN_ID="$run_id"
    export OUT_DIR="$OUT_DIR"
    export MASK_SCHEDULE="$mask_schedule"
    export TRAIN_TIMESTEP_SAMPLING="$timestep_sampling"
    export PARAMETERIZATION="$parameterization"
    export SELF_CONDITIONING="$self_conditioning"
    export LOSS_REWEIGHTING="$loss_reweighting"
    "$PYTHON_BIN" train_diffusion.py
  ) >>"$RUNNER_LOG" 2>&1; then
    log "Completed $phase run_id=$run_id"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$phase" "$run_id" "ok" "$(basename "$config_path")" "$mask_schedule" "$timestep_sampling" \
      "$parameterization" "$self_conditioning" "$loss_reweighting" "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
    return 0
  fi

  log "FAILED $phase run_id=$run_id"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$phase" "$run_id" "failed" "$(basename "$config_path")" "$mask_schedule" "$timestep_sampling" \
    "$parameterization" "$self_conditioning" "$loss_reweighting" "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
  return 1
}

run_full_eval() {
  local phase="$1"
  local run_id="$2"
  local config_path="$3"
  local checkpoint_path="$4"
  local eval_log_path="$OUT_DIR/${run_id}_diffusion_last_mlx_full_eval.txt"

  log "Starting $phase full eval for run_id=$run_id checkpoint=$(basename "$checkpoint_path")"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN: skipping diffusion_eval.py for $run_id"
    return 0
  fi

  if (
    set -a
    source "$config_path"
    set +a
    export OUT_DIR="$OUT_DIR"
    export VAL_MAX_TOKENS=0
    "$PYTHON_BIN" diffusion_eval.py --checkpoint "$checkpoint_path"
  ) >>"$RUNNER_LOG" 2>&1; then
    log "Completed $phase full eval for run_id=$run_id log=$(basename "$eval_log_path")"
    return 0
  fi

  log "FAILED $phase full eval for run_id=$run_id"
  return 1
}

FAILED=0
BASELINE_RUN_ID="week3_lock_baseline_${BATCH_ID}"
BASELINE_CONFIG="$ROOT_DIR/configs/diffusion_local.env"
WEEK3_CONFIG="$ROOT_DIR/configs/diffusion_local.env"

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"

run_with_config "baseline_lock" "$BASELINE_RUN_ID" "$BASELINE_CONFIG" "cosine" "random" "x0" "0" "none" || FAILED=1

BASELINE_CKPT="$OUT_DIR/${BASELINE_RUN_ID}_diffusion_last_mlx.npz"
if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: skipping baseline full eval checkpoint existence check"
elif [[ -f "$BASELINE_CKPT" ]]; then
  run_full_eval "baseline_lock" "$BASELINE_RUN_ID" "$BASELINE_CONFIG" "$BASELINE_CKPT" || FAILED=1
else
  log "Skipping baseline full eval because checkpoint is missing: $BASELINE_CKPT"
  FAILED=1
fi

declare -a SCHEDULES=("linear" "cosine" "loglinear")
declare -a TIMESTEP_SAMPLINGS=("random" "cyclic")

for schedule in "${SCHEDULES[@]}"; do
  for timestep_sampling in "${TIMESTEP_SAMPLINGS[@]}"; do
    run_id="week3_stageA_${schedule}_${timestep_sampling}_${BATCH_ID}"
    run_with_config "stage_a" "$run_id" "$WEEK3_CONFIG" "$schedule" "$timestep_sampling" "x0" "0" "none" || FAILED=1
  done
done

if [[ "$FAILED" == "0" ]]; then
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

log "Batch completed with failures. Summary: $SUMMARY_TSV"
exit 1
