#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_d_param_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/diffusion_week3_local.env}"
SCREEN_ITERATIONS="${SCREEN_ITERATIONS:-1500}"
LONG_ITERATIONS="${LONG_ITERATIONS:-3000}"
XTMINUS1_MARGIN="${XTMINUS1_MARGIN:-0.02}"
FORCE_LONG_XTMINUS1="${FORCE_LONG_XTMINUS1:-0}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-0}"

mkdir -p "$OUT_DIR"
touch "$RUNNER_LOG"

printf "phase\trun_id\tstatus\tconfig\tmask_schedule\ttrain_timestep_sampling\tparameterization\tself_conditioning\tloss_reweighting\titerations\tcheckpoint\tbest_checkpoint\tlog_path\n" > "$SUMMARY_TSV"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  local msg="$1"
  printf "[%s] %s\n" "$(timestamp)" "$msg" | tee -a "$RUNNER_LOG"
}

load_base_recipe_from_config() {
  set -a
  source "$CONFIG_PATH"
  set +a
  BASE_MASK_SCHEDULE="${MASK_SCHEDULE}"
  BASE_TRAIN_TIMESTEP_SAMPLING="${TRAIN_TIMESTEP_SAMPLING}"
  BASE_SELF_CONDITIONING="${SELF_CONDITIONING}"
  BASE_LOSS_REWEIGHTING="${LOSS_REWEIGHTING}"
  BASE_PARAMETERIZATION="${PARAMETERIZATION}"
  BASE_CONFIG_RUN_ID="${RUN_ID}"
}

assert_recipe_matches_log() {
  local log_path="$1"
  local expected_run_id="$2"
  local expected_mask_schedule="$3"
  local expected_timestep_sampling="$4"
  local expected_parameterization="$5"
  local expected_self_conditioning="$6"
  local expected_loss_reweighting="$7"

  LOG_PATH="$log_path" \
  EXPECTED_RUN_ID="$expected_run_id" \
  EXPECTED_MASK_SCHEDULE="$expected_mask_schedule" \
  EXPECTED_TIMESTEP="$expected_timestep_sampling" \
  EXPECTED_PARAMETERIZATION="$expected_parameterization" \
  EXPECTED_SELF_CONDITIONING="$expected_self_conditioning" \
  EXPECTED_LOSS_REWEIGHTING="$expected_loss_reweighting" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os
import re
import sys

log_path = Path(os.environ["LOG_PATH"])
text = log_path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"config_json:(\{.*?\})", text)
if not matches:
    raise SystemExit(f"No config_json payload found in {log_path}")
payload = json.loads(matches[-1])

expected = {
    "run_id": os.environ["EXPECTED_RUN_ID"],
    "mask_schedule": os.environ["EXPECTED_MASK_SCHEDULE"],
    "train_timestep_sampling": os.environ["EXPECTED_TIMESTEP"],
    "parameterization": os.environ["EXPECTED_PARAMETERIZATION"],
    "self_conditioning": os.environ["EXPECTED_SELF_CONDITIONING"] == "1",
    "loss_reweighting": os.environ["EXPECTED_LOSS_REWEIGHTING"],
}

for key, expected_value in expected.items():
    actual_value = payload.get(key)
    if actual_value != expected_value:
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

best_metric_for_run() {
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
metric = payload.get("best_metric_value")
if metric is None:
    raise SystemExit(f"No best metric recorded for {run_id}")
print(metric)
PY
}

run_train() {
  local phase="$1"
  local run_id="$2"
  local parameterization="$3"
  local iterations="$4"

  local train_log_path="$OUT_DIR/${run_id}_diffusion.txt"
  local last_ckpt="$OUT_DIR/${run_id}_diffusion_last_mlx.npz"
  local best_ckpt="$OUT_DIR/${run_id}_diffusion_best_mlx.npz"

  log "Starting $phase run_id=$run_id schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING parameterization=$parameterization self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING iterations=$iterations"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN: skipping train_diffusion.py for $run_id"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$phase" "$run_id" "dry-run" "$(basename "$CONFIG_PATH")" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$parameterization" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$iterations" \
      "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
    return 0
  fi

  if (
    set -a
    source "$CONFIG_PATH"
    set +a
    export RUN_ID="$run_id"
    export OUT_DIR="$OUT_DIR"
    export ITERATIONS="$iterations"
    export MASK_SCHEDULE="$BASE_MASK_SCHEDULE"
    export TRAIN_TIMESTEP_SAMPLING="$BASE_TRAIN_TIMESTEP_SAMPLING"
    export PARAMETERIZATION="$parameterization"
    export SELF_CONDITIONING="$BASE_SELF_CONDITIONING"
    export LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING"
    "$PYTHON_BIN" train_diffusion.py
  ) >>"$RUNNER_LOG" 2>&1; then
    if ! assert_recipe_matches_log \
      "$train_log_path" \
      "$run_id" \
      "$BASE_MASK_SCHEDULE" \
      "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$parameterization" \
      "$BASE_SELF_CONDITIONING" \
      "$BASE_LOSS_REWEIGHTING" >>"$RUNNER_LOG" 2>&1; then
      log "FAILED $phase run_id=$run_id due to recipe verification mismatch"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$phase" "$run_id" "failed" "$(basename "$CONFIG_PATH")" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
        "$parameterization" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$iterations" \
        "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
      return 1
    fi
    log "Completed $phase run_id=$run_id"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$phase" "$run_id" "ok" "$(basename "$CONFIG_PATH")" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$parameterization" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$iterations" \
      "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
    return 0
  fi

  log "FAILED $phase run_id=$run_id"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$phase" "$run_id" "failed" "$(basename "$CONFIG_PATH")" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
    "$parameterization" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$iterations" \
    "$last_ckpt" "$best_ckpt" "$train_log_path" >> "$SUMMARY_TSV"
  return 1
}

run_full_eval() {
  local phase="$1"
  local run_id="$2"
  local checkpoint_path="$3"
  local parameterization="$4"

  log "Starting $phase full eval for run_id=$run_id checkpoint=$(basename "$checkpoint_path")"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN: skipping diffusion_eval.py for $run_id"
    return 0
  fi

  if (
    set -a
    source "$CONFIG_PATH"
    set +a
    export OUT_DIR="$OUT_DIR"
    export VAL_MAX_TOKENS=0
    export MASK_SCHEDULE="$BASE_MASK_SCHEDULE"
    export TRAIN_TIMESTEP_SAMPLING="$BASE_TRAIN_TIMESTEP_SAMPLING"
    export PARAMETERIZATION="$parameterization"
    export SELF_CONDITIONING="$BASE_SELF_CONDITIONING"
    export LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING"
    "$PYTHON_BIN" diffusion_eval.py --checkpoint "$checkpoint_path"
  ) >>"$RUNNER_LOG" 2>&1; then
    log "Completed $phase full eval for run_id=$run_id"
    return 0
  fi

  log "FAILED $phase full eval for run_id=$run_id"
  return 1
}

load_base_recipe_from_config

FAILED=0
X0_RUN_ID="week3_p3_x0_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_${BATCH_ID}"
XTMINUS1_RUN_ID="week3_p3_xtminus1_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_${BATCH_ID}"
XTMINUS1_LONG_RUN_ID="week3_p3_xtminus1_long_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_${BATCH_ID}"

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"
log "Base recipe source: $(basename "$CONFIG_PATH")"
log "Base recipe: run_id=${BASE_CONFIG_RUN_ID:-from-config} schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING"

run_train "stage_d_p3_screen" "$X0_RUN_ID" "x0" "$SCREEN_ITERATIONS" || FAILED=1
run_train "stage_d_p3_screen" "$XTMINUS1_RUN_ID" "xtminus1" "$SCREEN_ITERATIONS" || FAILED=1

if [[ "$FAILED" != "0" ]]; then
  log "Batch completed with failures during screen runs. Summary: $SUMMARY_TSV"
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: skipping metric-based xtminus1 long-run decision"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

X0_BEST_METRIC="$(best_metric_for_run "$X0_RUN_ID" 2>>"$RUNNER_LOG")" || {
  log "Could not determine the best metric for $X0_RUN_ID"
  exit 1
}
XTMINUS1_BEST_METRIC="$(best_metric_for_run "$XTMINUS1_RUN_ID" 2>>"$RUNNER_LOG")" || {
  log "Could not determine the best metric for $XTMINUS1_RUN_ID"
  exit 1
}

log "P3 screen results: x0_best=$X0_BEST_METRIC xtminus1_best=$XTMINUS1_BEST_METRIC margin=$XTMINUS1_MARGIN"

SHOULD_RUN_LONG=0
if [[ "$FORCE_LONG_XTMINUS1" == "1" ]]; then
  SHOULD_RUN_LONG=1
else
  if X0_BEST_METRIC="$X0_BEST_METRIC" XTMINUS1_BEST_METRIC="$XTMINUS1_BEST_METRIC" XTMINUS1_MARGIN="$XTMINUS1_MARGIN" "$PYTHON_BIN" - <<'PY'
import os
import sys
x0 = float(os.environ["X0_BEST_METRIC"])
xt = float(os.environ["XTMINUS1_BEST_METRIC"])
margin = float(os.environ["XTMINUS1_MARGIN"])
sys.exit(0 if xt <= x0 + margin else 1)
PY
  then
    SHOULD_RUN_LONG=1
  fi
fi

if [[ "$SHOULD_RUN_LONG" != "1" ]]; then
  log "xtminus1 was not within margin of x0; skipping the 3000-step follow-up"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

log "xtminus1 qualified for the long follow-up"
run_train "stage_d_p3_long" "$XTMINUS1_LONG_RUN_ID" "xtminus1" "$LONG_ITERATIONS" || {
  log "Batch completed with failures during the long xtminus1 run. Summary: $SUMMARY_TSV"
  exit 1
}

if [[ "$RUN_FULL_EVAL" == "1" ]]; then
  XTMINUS1_LONG_CHECKPOINT="$(best_checkpoint_for_run "$XTMINUS1_LONG_RUN_ID" 2>>"$RUNNER_LOG")" || {
    log "Could not determine the checkpoint for the xtminus1 long run"
    exit 1
  }
  run_full_eval "stage_d_p3_long" "$XTMINUS1_LONG_RUN_ID" "$XTMINUS1_LONG_CHECKPOINT" "xtminus1" || exit 1
fi

log "Batch completed successfully. Summary: $SUMMARY_TSV"
exit 0
