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

REQUESTED_RUN_ID="${RUN_ID:-}"

BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_h_continue_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/diffusion_scale.env}"
CONTINUE_ITERATIONS="${CONTINUE_ITERATIONS:-7000}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_bpb}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
INIT_CHECKPOINT_PATH="${INIT_CHECKPOINT_PATH:-$ROOT_DIR/logs/week3_stage_g_scale_20260412_154123/diffusion_week3_scale_diffusion_best_mlx.npz}"

mkdir -p "$OUT_DIR"
touch "$RUNNER_LOG"

printf "phase\trun_id\tstatus\tconfig\tmask_schedule\ttrain_timestep_sampling\tparameterization\tself_conditioning\tloss_reweighting\tnum_diffusion_steps\tmin_mask_rate\tmax_mask_rate\tlearning_rate\tweight_decay\tbeta2\tgrad_clip_norm\twarmup_steps\titerations\tinit_checkpoint\tearly_stop_patience\tearly_stop_metric\tearly_stop_min_delta\tcheckpoint\tbest_checkpoint\tlog_path\n" > "$SUMMARY_TSV"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  local msg="$1"
  printf "[%s] %s\n" "$(timestamp)" "$msg" | tee -a "$RUNNER_LOG"
}

append_summary_row() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$(basename "$CONFIG_PATH")" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" "${19}" "${20}" "${21}" "${22}" "${23}" "${24}" >> "$SUMMARY_TSV"
}

float_tag() {
  VALUE="$1" "$PYTHON_BIN" - <<'PY'
import math
import os

value = float(os.environ["VALUE"])
if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
    text = str(int(round(value)))
else:
    text = format(value, ".12g")
text = text.replace("-", "m").replace(".", "p")
print(text)
PY
}

load_recipe_from_config() {
  set -a
  source "$CONFIG_PATH"
  set +a
  BASE_MASK_SCHEDULE="${MASK_SCHEDULE}"
  BASE_TRAIN_TIMESTEP_SAMPLING="${TRAIN_TIMESTEP_SAMPLING}"
  BASE_SELF_CONDITIONING="${SELF_CONDITIONING}"
  BASE_LOSS_REWEIGHTING="${LOSS_REWEIGHTING}"
  BASE_PARAMETERIZATION="${PARAMETERIZATION}"
  BASE_NUM_DIFFUSION_STEPS="${NUM_DIFFUSION_STEPS}"
  BASE_MIN_MASK_RATE="${MIN_MASK_RATE:-0.0}"
  BASE_MAX_MASK_RATE="${MAX_MASK_RATE:-1.0}"
  BASE_LEARNING_RATE="${LEARNING_RATE}"
  BASE_WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
  BASE_BETA2="${BETA2:-0.95}"
  BASE_GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
  BASE_WARMUP_STEPS="${WARMUP_STEPS:-5}"
}

assert_recipe_matches_log() {
  local log_path="$1"
  local expected_run_id="$2"

  LOG_PATH="$log_path" \
  EXPECTED_RUN_ID="$expected_run_id" \
  EXPECTED_MASK_SCHEDULE="$BASE_MASK_SCHEDULE" \
  EXPECTED_TIMESTEP="$BASE_TRAIN_TIMESTEP_SAMPLING" \
  EXPECTED_PARAMETERIZATION="$BASE_PARAMETERIZATION" \
  EXPECTED_SELF_CONDITIONING="$BASE_SELF_CONDITIONING" \
  EXPECTED_LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING" \
  EXPECTED_NUM_DIFFUSION_STEPS="$BASE_NUM_DIFFUSION_STEPS" \
  EXPECTED_MIN_MASK_RATE="$BASE_MIN_MASK_RATE" \
  EXPECTED_MAX_MASK_RATE="$BASE_MAX_MASK_RATE" \
  EXPECTED_LEARNING_RATE="$BASE_LEARNING_RATE" \
  EXPECTED_WEIGHT_DECAY="$BASE_WEIGHT_DECAY" \
  EXPECTED_BETA2="$BASE_BETA2" \
  EXPECTED_GRAD_CLIP_NORM="$BASE_GRAD_CLIP_NORM" \
  EXPECTED_WARMUP_STEPS="$BASE_WARMUP_STEPS" \
  EXPECTED_INIT_CHECKPOINT="$INIT_CHECKPOINT_PATH" \
  EXPECTED_EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
  EXPECTED_EARLY_STOP_METRIC="$EARLY_STOP_METRIC" \
  EXPECTED_EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import math
import os
import re

log_path = Path(os.environ["LOG_PATH"])
text = log_path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"config_json:(\{.*?\})", text)
if not matches:
    raise SystemExit(f"No config_json payload found in {log_path}")
payload = json.loads(matches[-1])

expected_exact = {
    "run_id": os.environ["EXPECTED_RUN_ID"],
    "mask_schedule": os.environ["EXPECTED_MASK_SCHEDULE"],
    "train_timestep_sampling": os.environ["EXPECTED_TIMESTEP"],
    "parameterization": os.environ["EXPECTED_PARAMETERIZATION"],
    "self_conditioning": os.environ["EXPECTED_SELF_CONDITIONING"] == "1",
    "loss_reweighting": os.environ["EXPECTED_LOSS_REWEIGHTING"],
    "num_diffusion_steps": int(os.environ["EXPECTED_NUM_DIFFUSION_STEPS"]),
    "warmup_steps": int(os.environ["EXPECTED_WARMUP_STEPS"]),
    "init_checkpoint": os.environ["EXPECTED_INIT_CHECKPOINT"],
    "early_stop_patience": int(os.environ["EXPECTED_EARLY_STOP_PATIENCE"]),
    "early_stop_metric": os.environ["EXPECTED_EARLY_STOP_METRIC"],
}

for key, expected_value in expected_exact.items():
    actual_value = payload.get(key)
    if actual_value != expected_value:
        raise SystemExit(
            f"Recipe mismatch for {key}: expected {expected_value!r}, got {actual_value!r}"
        )

for key, env_name in (
    ("min_mask_rate", "EXPECTED_MIN_MASK_RATE"),
    ("max_mask_rate", "EXPECTED_MAX_MASK_RATE"),
    ("learning_rate", "EXPECTED_LEARNING_RATE"),
    ("weight_decay", "EXPECTED_WEIGHT_DECAY"),
    ("beta2", "EXPECTED_BETA2"),
    ("grad_clip_norm", "EXPECTED_GRAD_CLIP_NORM"),
    ("early_stop_min_delta", "EXPECTED_EARLY_STOP_MIN_DELTA"),
):
    actual_value = float(payload.get(key))
    expected_value = float(os.environ[env_name])
    if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"Recipe mismatch for {key}: expected {expected_value!r}, got {actual_value!r}"
        )
PY
}

best_checkpoint_for_run() {
  local run_id="$1"
  OUT_DIR="$OUT_DIR" RUN_ID="$run_id" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

out_dir = Path(os.environ["OUT_DIR"])
run_id = os.environ["RUN_ID"]
manifest_path = out_dir / f"{run_id}_diffusion_manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"Missing manifest for {run_id}")

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
checkpoint = payload.get("best_checkpoint") or payload.get("last_checkpoint")
if not checkpoint:
    raise SystemExit(f"No checkpoint recorded for {run_id}")
print(checkpoint)
PY
}

load_recipe_from_config

RUN_ID="${REQUESTED_RUN_ID:-week3_p7_continue_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_param${BASE_PARAMETERIZATION}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_steps${BASE_NUM_DIFFUSION_STEPS}_min$(float_tag "$BASE_MIN_MASK_RATE")_max$(float_tag "$BASE_MAX_MASK_RATE")_lr$(float_tag "$BASE_LEARNING_RATE")_wd$(float_tag "$BASE_WEIGHT_DECAY")_b2$(float_tag "$BASE_BETA2")_clip$(float_tag "$BASE_GRAD_CLIP_NORM")_wu${BASE_WARMUP_STEPS}_init$(basename "$INIT_CHECKPOINT_PATH" .npz)_es${EARLY_STOP_PATIENCE}_${BATCH_ID}}"

TRAIN_LOG_PATH="$OUT_DIR/${RUN_ID}_diffusion.txt"
LAST_CKPT="$OUT_DIR/${RUN_ID}_diffusion_last_mlx.npz"
BEST_CKPT="$OUT_DIR/${RUN_ID}_diffusion_best_mlx.npz"

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"
log "Running P7 weights-only continuation with run_id=$RUN_ID"
log "Recipe: schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING parameterization=$BASE_PARAMETERIZATION self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING diffusion_steps=$BASE_NUM_DIFFUSION_STEPS min_mask_rate=$BASE_MIN_MASK_RATE max_mask_rate=$BASE_MAX_MASK_RATE lr=$BASE_LEARNING_RATE wd=$BASE_WEIGHT_DECAY beta2=$BASE_BETA2 grad_clip_norm=$BASE_GRAD_CLIP_NORM warmup_steps=$BASE_WARMUP_STEPS iterations=$CONTINUE_ITERATIONS"
log "Continuation: init_checkpoint=$INIT_CHECKPOINT_PATH early_stop_patience=$EARLY_STOP_PATIENCE early_stop_metric=$EARLY_STOP_METRIC early_stop_min_delta=$EARLY_STOP_MIN_DELTA"

if [[ ! -f "$INIT_CHECKPOINT_PATH" ]]; then
  log "Missing init checkpoint: $INIT_CHECKPOINT_PATH"
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: skipping train_diffusion.py"
  append_summary_row \
    "stage_h_p7_continue" "$RUN_ID" "dry-run" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
    "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$BASE_NUM_DIFFUSION_STEPS" "$BASE_MIN_MASK_RATE" "$BASE_MAX_MASK_RATE" \
    "$BASE_LEARNING_RATE" "$BASE_WEIGHT_DECAY" "$BASE_BETA2" "$BASE_GRAD_CLIP_NORM" "$BASE_WARMUP_STEPS" "$CONTINUE_ITERATIONS" \
    "$INIT_CHECKPOINT_PATH" "$EARLY_STOP_PATIENCE" "$EARLY_STOP_METRIC" "$EARLY_STOP_MIN_DELTA" \
    "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH"
else
  if (
    set -a
    source "$CONFIG_PATH"
    set +a
    export RUN_ID="$RUN_ID"
    export OUT_DIR="$OUT_DIR"
    export ITERATIONS="$CONTINUE_ITERATIONS"
    export INIT_CHECKPOINT="$INIT_CHECKPOINT_PATH"
    export EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE"
    export EARLY_STOP_METRIC="$EARLY_STOP_METRIC"
    export EARLY_STOP_MIN_DELTA="$EARLY_STOP_MIN_DELTA"
    "$PYTHON_BIN" train_diffusion.py
  ) >>"$RUNNER_LOG" 2>&1; then
    if ! assert_recipe_matches_log "$TRAIN_LOG_PATH" "$RUN_ID" >>"$RUNNER_LOG" 2>&1; then
      log "FAILED stage_h_p7_continue run_id=$RUN_ID due to recipe verification mismatch"
      append_summary_row \
        "stage_h_p7_continue" "$RUN_ID" "failed" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
        "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$BASE_NUM_DIFFUSION_STEPS" "$BASE_MIN_MASK_RATE" "$BASE_MAX_MASK_RATE" \
        "$BASE_LEARNING_RATE" "$BASE_WEIGHT_DECAY" "$BASE_BETA2" "$BASE_GRAD_CLIP_NORM" "$BASE_WARMUP_STEPS" "$CONTINUE_ITERATIONS" \
        "$INIT_CHECKPOINT_PATH" "$EARLY_STOP_PATIENCE" "$EARLY_STOP_METRIC" "$EARLY_STOP_MIN_DELTA" \
        "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH"
      exit 1
    fi
    log "Completed stage_h_p7_continue run_id=$RUN_ID"
    append_summary_row \
      "stage_h_p7_continue" "$RUN_ID" "ok" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$BASE_NUM_DIFFUSION_STEPS" "$BASE_MIN_MASK_RATE" "$BASE_MAX_MASK_RATE" \
      "$BASE_LEARNING_RATE" "$BASE_WEIGHT_DECAY" "$BASE_BETA2" "$BASE_GRAD_CLIP_NORM" "$BASE_WARMUP_STEPS" "$CONTINUE_ITERATIONS" \
      "$INIT_CHECKPOINT_PATH" "$EARLY_STOP_PATIENCE" "$EARLY_STOP_METRIC" "$EARLY_STOP_MIN_DELTA" \
      "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH"
  else
    log "FAILED stage_h_p7_continue run_id=$RUN_ID"
    append_summary_row \
      "stage_h_p7_continue" "$RUN_ID" "failed" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$BASE_NUM_DIFFUSION_STEPS" "$BASE_MIN_MASK_RATE" "$BASE_MAX_MASK_RATE" \
      "$BASE_LEARNING_RATE" "$BASE_WEIGHT_DECAY" "$BASE_BETA2" "$BASE_GRAD_CLIP_NORM" "$BASE_WARMUP_STEPS" "$CONTINUE_ITERATIONS" \
      "$INIT_CHECKPOINT_PATH" "$EARLY_STOP_PATIENCE" "$EARLY_STOP_METRIC" "$EARLY_STOP_MIN_DELTA" \
      "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH"
    exit 1
  fi
fi

if [[ "$RUN_FULL_EVAL" != "1" ]]; then
  log "Skipping full eval because RUN_FULL_EVAL=$RUN_FULL_EVAL"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: skipping diffusion_eval.py"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

if ! BEST_CHECKPOINT_PATH="$(best_checkpoint_for_run "$RUN_ID" 2>>"$RUNNER_LOG")"; then
  log "Could not determine the checkpoint to full-eval for run_id=$RUN_ID"
  exit 1
fi

log "Starting stage_h_p7_continue full eval for run_id=$RUN_ID checkpoint=$(basename "$BEST_CHECKPOINT_PATH")"
if (
  set -a
  source "$CONFIG_PATH"
  set +a
  export OUT_DIR="$OUT_DIR"
  export VAL_MAX_TOKENS=0
  "$PYTHON_BIN" diffusion_eval.py --checkpoint "$BEST_CHECKPOINT_PATH"
) >>"$RUNNER_LOG" 2>&1; then
  log "Completed stage_h_p7_continue full eval for run_id=$RUN_ID"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

log "FAILED stage_h_p7_continue full eval for run_id=$RUN_ID"
exit 1
