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
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_e_process_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/diffusion_local.env}"
SCREEN_ITERATIONS="${SCREEN_ITERATIONS:-1500}"
LONG_ITERATIONS="${LONG_ITERATIONS:-3000}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"
DIFFUSION_STEPS_LIST="${DIFFUSION_STEPS_LIST:-16,32,64}"
MASK_BOUNDS_LIST="${MASK_BOUNDS_LIST:-0.0:1.0,0.05:1.0,0.0:0.95}"
PREVIOUS_SOTA_1500="${PREVIOUS_SOTA_1500:-}"
PREVIOUS_SOTA_1500_MANIFEST="${PREVIOUS_SOTA_1500_MANIFEST:-$ROOT_DIR/logs/week3_stage_a_20260408_143331/week3_stageA_linear_cyclic_20260408_143331_diffusion_manifest.json}"
PROMOTION_MARGIN="${PROMOTION_MARGIN:-0.0}"

mkdir -p "$OUT_DIR"
touch "$RUNNER_LOG"

printf "phase\trun_id\tstatus\tconfig\tmask_schedule\ttrain_timestep_sampling\tparameterization\tself_conditioning\tloss_reweighting\tnum_diffusion_steps\tmin_mask_rate\tmax_mask_rate\titerations\tcheckpoint\tbest_checkpoint\tlog_path\tmanifest_path\n" > "$SUMMARY_TSV"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  local msg="$1"
  printf "[%s] %s\n" "$(timestamp)" "$msg" | tee -a "$RUNNER_LOG"
}

append_summary_row() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$(basename "$CONFIG_PATH")" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" >> "$SUMMARY_TSV"
}

float_tag() {
  local value="$1"
  value="${value//- /}"
  value="${value//-/m}"
  value="${value//./p}"
  echo "$value"
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
  BASE_NUM_DIFFUSION_STEPS="${NUM_DIFFUSION_STEPS}"
  BASE_MIN_MASK_RATE="${MIN_MASK_RATE:-0.0}"
  BASE_MAX_MASK_RATE="${MAX_MASK_RATE:-1.0}"
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
  local expected_num_diffusion_steps="$8"
  local expected_min_mask_rate="$9"
  local expected_max_mask_rate="${10}"

  LOG_PATH="$log_path" \
  EXPECTED_RUN_ID="$expected_run_id" \
  EXPECTED_MASK_SCHEDULE="$expected_mask_schedule" \
  EXPECTED_TIMESTEP="$expected_timestep_sampling" \
  EXPECTED_PARAMETERIZATION="$expected_parameterization" \
  EXPECTED_SELF_CONDITIONING="$expected_self_conditioning" \
  EXPECTED_LOSS_REWEIGHTING="$expected_loss_reweighting" \
  EXPECTED_NUM_DIFFUSION_STEPS="$expected_num_diffusion_steps" \
  EXPECTED_MIN_MASK_RATE="$expected_min_mask_rate" \
  EXPECTED_MAX_MASK_RATE="$expected_max_mask_rate" \
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

expected = {
    "run_id": os.environ["EXPECTED_RUN_ID"],
    "mask_schedule": os.environ["EXPECTED_MASK_SCHEDULE"],
    "train_timestep_sampling": os.environ["EXPECTED_TIMESTEP"],
    "parameterization": os.environ["EXPECTED_PARAMETERIZATION"],
    "self_conditioning": os.environ["EXPECTED_SELF_CONDITIONING"] == "1",
    "loss_reweighting": os.environ["EXPECTED_LOSS_REWEIGHTING"],
    "num_diffusion_steps": int(os.environ["EXPECTED_NUM_DIFFUSION_STEPS"]),
}

for key, expected_value in expected.items():
    actual_value = payload.get(key)
    if actual_value != expected_value:
        raise SystemExit(
            f"Recipe mismatch for {key}: expected {expected_value!r}, got {actual_value!r}"
        )

for key, env_name in (
    ("min_mask_rate", "EXPECTED_MIN_MASK_RATE"),
    ("max_mask_rate", "EXPECTED_MAX_MASK_RATE"),
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
  OUT_DIR="$OUT_DIR" RUN_ID="$run_id" SUMMARY_TSV="$SUMMARY_TSV" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import json
import os

out_dir = Path(os.environ["OUT_DIR"])
run_id = os.environ["RUN_ID"]
manifest_path = out_dir / f"{run_id}_diffusion_manifest.json"
if not manifest_path.exists():
    summary_tsv = Path(os.environ["SUMMARY_TSV"])
    if summary_tsv.exists():
        with summary_tsv.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("run_id") != run_id:
                    continue
                candidate = row.get("manifest_path", "").strip()
                if candidate:
                    manifest_path = Path(candidate)
                    break
if not manifest_path.exists():
    raise SystemExit(f"Missing manifest for {run_id}")

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
checkpoint = payload.get("best_checkpoint") or payload.get("last_checkpoint")
if not checkpoint:
    raise SystemExit(f"No checkpoint recorded for {run_id}")
print(checkpoint)
PY
}

find_existing_completed_run() {
  local phase="$1"
  local num_diffusion_steps="$2"
  local min_mask_rate="$3"
  local max_mask_rate="$4"
  local iterations="$5"

  ROOT_DIR="$ROOT_DIR" \
  CURRENT_SUMMARY="$SUMMARY_TSV" \
  EXPECTED_PHASE="$phase" \
  EXPECTED_MASK_SCHEDULE="$BASE_MASK_SCHEDULE" \
  EXPECTED_TIMESTEP="$BASE_TRAIN_TIMESTEP_SAMPLING" \
  EXPECTED_PARAMETERIZATION="$BASE_PARAMETERIZATION" \
  EXPECTED_SELF_CONDITIONING="$BASE_SELF_CONDITIONING" \
  EXPECTED_LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING" \
  EXPECTED_NUM_DIFFUSION_STEPS="$num_diffusion_steps" \
  EXPECTED_MIN_MASK_RATE="$min_mask_rate" \
  EXPECTED_MAX_MASK_RATE="$max_mask_rate" \
  EXPECTED_ITERATIONS="$iterations" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import json
import math
import os

root_dir = Path(os.environ["ROOT_DIR"])
current_summary = Path(os.environ["CURRENT_SUMMARY"]).resolve()

expected_phase = os.environ["EXPECTED_PHASE"]
expected_exact = {
    "mask_schedule": os.environ["EXPECTED_MASK_SCHEDULE"],
    "train_timestep_sampling": os.environ["EXPECTED_TIMESTEP"],
    "parameterization": os.environ["EXPECTED_PARAMETERIZATION"],
    "self_conditioning": os.environ["EXPECTED_SELF_CONDITIONING"],
    "loss_reweighting": os.environ["EXPECTED_LOSS_REWEIGHTING"],
    "num_diffusion_steps": os.environ["EXPECTED_NUM_DIFFUSION_STEPS"],
    "iterations": os.environ["EXPECTED_ITERATIONS"],
}
expected_float = {
    "min_mask_rate": float(os.environ["EXPECTED_MIN_MASK_RATE"]),
    "max_mask_rate": float(os.environ["EXPECTED_MAX_MASK_RATE"]),
}

best = None
for summary_path in sorted(root_dir.glob("logs/week3_stage_e_process_*/summary.tsv"), reverse=True):
    if summary_path.resolve() == current_summary:
        continue
    with summary_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("phase") != expected_phase:
                continue
            if row.get("status") not in {"ok", "reused"}:
                continue
            matches = True
            for key, expected_value in expected_exact.items():
                if row.get(key, "") != expected_value:
                    matches = False
                    break
            if not matches:
                continue
            for key, expected_value in expected_float.items():
                try:
                    actual = float(row.get(key, "nan"))
                except ValueError:
                    matches = False
                    break
                if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-12):
                    matches = False
                    break
            if not matches:
                continue

            run_id = row["run_id"]
            manifest_value = row.get("manifest_path", "").strip()
            manifest_path = Path(manifest_value) if manifest_value else summary_path.parent / f"{run_id}_diffusion_manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            metric = payload.get("best_metric_value")
            if metric is None:
                continue
            candidate = (
                float(metric),
                run_id,
                str(manifest_path),
                row.get("checkpoint", ""),
                row.get("best_checkpoint", ""),
                row.get("log_path", ""),
                str(summary_path),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate

if best is None:
    raise SystemExit(1)

for value in best[1:]:
    print(value)
PY
}

previous_sota_1500_metric() {
  if [[ -n "$PREVIOUS_SOTA_1500" ]]; then
    printf "%s\n" "$PREVIOUS_SOTA_1500"
    return 0
  fi

  MANIFEST_PATH="$PREVIOUS_SOTA_1500_MANIFEST" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

manifest_path = Path(os.environ["MANIFEST_PATH"])
if not manifest_path.exists():
    raise SystemExit(f"Missing previous SOTA manifest: {manifest_path}")

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
metric = payload.get("best_metric_value")
if metric is None:
    raise SystemExit(f"No best_metric_value recorded in {manifest_path}")
print(metric)
PY
}

validate_elbo_recipe() {
  local num_diffusion_steps="$1"
  local min_mask_rate="$2"
  local max_mask_rate="$3"

  NUM_DIFFUSION_STEPS_VALUE="$num_diffusion_steps" \
  MASK_SCHEDULE_VALUE="$BASE_MASK_SCHEDULE" \
  MIN_MASK_RATE_VALUE="$min_mask_rate" \
  MAX_MASK_RATE_VALUE="$max_mask_rate" \
  "$PYTHON_BIN" - <<'PY'
import os

from diffusion_objectives import build_mask_rates, validate_elbo_mask_rates

mask_rates = build_mask_rates(
    int(os.environ["NUM_DIFFUSION_STEPS_VALUE"]),
    os.environ["MASK_SCHEDULE_VALUE"],
    float(os.environ["MIN_MASK_RATE_VALUE"]),
    float(os.environ["MAX_MASK_RATE_VALUE"]),
)
validate_elbo_mask_rates(mask_rates)
PY
}

best_run_for_phase() {
  local phase_prefix="$1"
  SUMMARY_TSV="$SUMMARY_TSV" OUT_DIR="$OUT_DIR" PHASE_PREFIX="$phase_prefix" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import json
import os

summary_tsv = Path(os.environ["SUMMARY_TSV"])
out_dir = Path(os.environ["OUT_DIR"])
phase_prefix = os.environ["PHASE_PREFIX"]
best = None

with summary_tsv.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("status") not in {"ok", "reused"}:
            continue
        phase = row.get("phase", "")
        if not phase.startswith(phase_prefix):
            continue
        run_id = row["run_id"]
        manifest_value = row.get("manifest_path", "").strip()
        manifest_path = Path(manifest_value) if manifest_value else out_dir / f"{run_id}_diffusion_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metric = payload.get("best_metric_value")
        if metric is None:
            continue
        candidate = (
            float(metric),
            row["run_id"],
            row["num_diffusion_steps"],
            row["min_mask_rate"],
            row["max_mask_rate"],
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

if best is None:
    raise SystemExit(f"No successful runs found for phase prefix {phase_prefix!r}")

for value in best[1:]:
    print(value)
print(f"{best[0]:.10f}")
PY
}

run_train() {
  local phase="$1"
  local run_id="$2"
  local num_diffusion_steps="$3"
  local min_mask_rate="$4"
  local max_mask_rate="$5"
  local iterations="$6"

  local train_log_path="$OUT_DIR/${run_id}_diffusion.txt"
  local last_ckpt="$OUT_DIR/${run_id}_diffusion_last_mlx.npz"
  local best_ckpt="$OUT_DIR/${run_id}_diffusion_best_mlx.npz"
  local manifest_path="$OUT_DIR/${run_id}_diffusion_manifest.json"

  log "Starting $phase run_id=$run_id schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING parameterization=$BASE_PARAMETERIZATION self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING diffusion_steps=$num_diffusion_steps min_mask_rate=$min_mask_rate max_mask_rate=$max_mask_rate iterations=$iterations"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN: skipping train_diffusion.py for $run_id"
    append_summary_row \
      "$phase" "$run_id" "dry-run" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$num_diffusion_steps" "$min_mask_rate" "$max_mask_rate" "$iterations" \
      "$last_ckpt" "$best_ckpt" "$train_log_path" "$manifest_path"
    return 0
  fi

  EXISTING_INFO=()
  while IFS= read -r line; do
    EXISTING_INFO+=("$line")
  done < <(find_existing_completed_run "$phase" "$num_diffusion_steps" "$min_mask_rate" "$max_mask_rate" "$iterations" 2>>"$RUNNER_LOG") || true

  if [[ "${#EXISTING_INFO[@]}" -ge 6 ]]; then
    local reused_run_id="${EXISTING_INFO[0]}"
    local reused_manifest_path="${EXISTING_INFO[1]}"
    local reused_last_ckpt="${EXISTING_INFO[2]}"
    local reused_best_ckpt="${EXISTING_INFO[3]}"
    local reused_log_path="${EXISTING_INFO[4]}"
    local reused_summary_path="${EXISTING_INFO[5]}"
    log "Reusing prior completed $phase run_id=$reused_run_id from $(basename "$(dirname "$reused_summary_path")") for requested run_id=$run_id"
    append_summary_row \
      "$phase" "$run_id" "reused" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$num_diffusion_steps" "$min_mask_rate" "$max_mask_rate" "$iterations" \
      "$reused_last_ckpt" "$reused_best_ckpt" "$reused_log_path" "$reused_manifest_path"
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
    export PARAMETERIZATION="$BASE_PARAMETERIZATION"
    export SELF_CONDITIONING="$BASE_SELF_CONDITIONING"
    export LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING"
    export NUM_DIFFUSION_STEPS="$num_diffusion_steps"
    export MIN_MASK_RATE="$min_mask_rate"
    export MAX_MASK_RATE="$max_mask_rate"
    "$PYTHON_BIN" train_diffusion.py
  ) >>"$RUNNER_LOG" 2>&1; then
    if ! assert_recipe_matches_log \
      "$train_log_path" \
      "$run_id" \
      "$BASE_MASK_SCHEDULE" \
      "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" \
      "$BASE_SELF_CONDITIONING" \
      "$BASE_LOSS_REWEIGHTING" \
      "$num_diffusion_steps" \
      "$min_mask_rate" \
      "$max_mask_rate" >>"$RUNNER_LOG" 2>&1; then
      log "FAILED $phase run_id=$run_id due to recipe verification mismatch"
      append_summary_row \
        "$phase" "$run_id" "failed" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
        "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$num_diffusion_steps" "$min_mask_rate" "$max_mask_rate" "$iterations" \
        "$last_ckpt" "$best_ckpt" "$train_log_path" "$manifest_path"
      return 1
    fi
    log "Completed $phase run_id=$run_id"
    append_summary_row \
      "$phase" "$run_id" "ok" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$num_diffusion_steps" "$min_mask_rate" "$max_mask_rate" "$iterations" \
      "$last_ckpt" "$best_ckpt" "$train_log_path" "$manifest_path"
    return 0
  fi

  log "FAILED $phase run_id=$run_id"
  append_summary_row \
    "$phase" "$run_id" "failed" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
    "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$num_diffusion_steps" "$min_mask_rate" "$max_mask_rate" "$iterations" \
    "$last_ckpt" "$best_ckpt" "$train_log_path" "$manifest_path"
  return 1
}

run_full_eval() {
  local phase="$1"
  local run_id="$2"
  local checkpoint_path="$3"
  local num_diffusion_steps="$4"
  local min_mask_rate="$5"
  local max_mask_rate="$6"

  log "Starting $phase full eval for run_id=$run_id checkpoint=$(basename "$checkpoint_path") diffusion_steps=$num_diffusion_steps min_mask_rate=$min_mask_rate max_mask_rate=$max_mask_rate"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY RUN: skipping diffusion_eval.py for $run_id"
    return 0
  fi

  local full_eval_path="${checkpoint_path%.npz}_full_eval.txt"
  if [[ -f "$full_eval_path" ]]; then
    log "Reusing existing $phase full eval for run_id=$run_id from $full_eval_path"
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
    export PARAMETERIZATION="$BASE_PARAMETERIZATION"
    export SELF_CONDITIONING="$BASE_SELF_CONDITIONING"
    export LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING"
    export NUM_DIFFUSION_STEPS="$num_diffusion_steps"
    export MIN_MASK_RATE="$min_mask_rate"
    export MAX_MASK_RATE="$max_mask_rate"
    "$PYTHON_BIN" diffusion_eval.py --checkpoint "$checkpoint_path"
  ) >>"$RUNNER_LOG" 2>&1; then
    log "Completed $phase full eval for run_id=$run_id"
    return 0
  fi

  log "FAILED $phase full eval for run_id=$run_id"
  return 1
}

should_promote_screen_winner() {
  SCREEN_BEST="$1" PREVIOUS_SOTA="$2" PROMOTION_MARGIN="$3" "$PYTHON_BIN" - <<'PY'
import os
import sys

screen_best = float(os.environ["SCREEN_BEST"])
previous_sota = float(os.environ["PREVIOUS_SOTA"])
margin = float(os.environ["PROMOTION_MARGIN"])
threshold = previous_sota - margin
sys.exit(0 if screen_best < threshold else 1)
PY
}

load_base_recipe_from_config

FAILED=0

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"
log "Base recipe source: $(basename "$CONFIG_PATH")"
log "Base recipe: run_id=${BASE_CONFIG_RUN_ID:-from-config} schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING parameterization=$BASE_PARAMETERIZATION self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING diffusion_steps=$BASE_NUM_DIFFUSION_STEPS min_mask_rate=$BASE_MIN_MASK_RATE max_mask_rate=$BASE_MAX_MASK_RATE"
log "Screen plan: SCREEN_ITERATIONS=$SCREEN_ITERATIONS LONG_ITERATIONS=$LONG_ITERATIONS"

IFS=',' read -r -a STEP_VALUES <<< "$DIFFUSION_STEPS_LIST"
for steps in "${STEP_VALUES[@]}"; do
  steps="${steps// /}"
  [[ -z "$steps" ]] && continue
  RUN_ID="week3_p4_steps${steps}_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_param${BASE_PARAMETERIZATION}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_min$(float_tag "$BASE_MIN_MASK_RATE")_max$(float_tag "$BASE_MAX_MASK_RATE")_${BATCH_ID}"
  run_train "stage_e_p4_steps" "$RUN_ID" "$steps" "$BASE_MIN_MASK_RATE" "$BASE_MAX_MASK_RATE" "$SCREEN_ITERATIONS" || FAILED=1
done

if [[ "$FAILED" != "0" ]]; then
  log "Batch completed with failures during the diffusion-step sweep. Summary: $SUMMARY_TSV"
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  BEST_STEP_NUM_DIFFUSION_STEPS="$BASE_NUM_DIFFUSION_STEPS"
  log "DRY RUN: using base diffusion_steps=$BEST_STEP_NUM_DIFFUSION_STEPS for the mask-bound sweep"
else
  BEST_STEP_INFO=()
  while IFS= read -r line; do
    BEST_STEP_INFO+=("$line")
  done < <(best_run_for_phase "stage_e_p4_steps" 2>>"$RUNNER_LOG")

  if [[ "${#BEST_STEP_INFO[@]}" -lt 5 ]]; then
    log "Could not determine the best diffusion-step run from the completed sweep"
    exit 1
  fi

  BEST_STEP_RUN_ID="${BEST_STEP_INFO[0]}"
  BEST_STEP_NUM_DIFFUSION_STEPS="${BEST_STEP_INFO[1]}"
  BEST_STEP_MIN_MASK_RATE="${BEST_STEP_INFO[2]}"
  BEST_STEP_MAX_MASK_RATE="${BEST_STEP_INFO[3]}"
  BEST_STEP_METRIC="${BEST_STEP_INFO[4]}"
  log "Best diffusion-step run: run_id=$BEST_STEP_RUN_ID diffusion_steps=$BEST_STEP_NUM_DIFFUSION_STEPS min_mask_rate=$BEST_STEP_MIN_MASK_RATE max_mask_rate=$BEST_STEP_MAX_MASK_RATE best_val_bpb=$BEST_STEP_METRIC"
fi

IFS=',' read -r -a BOUND_VALUES <<< "$MASK_BOUNDS_LIST"
for bounds in "${BOUND_VALUES[@]}"; do
  bounds="${bounds// /}"
  [[ -z "$bounds" ]] && continue
  min_mask_rate="${bounds%%:*}"
  max_mask_rate="${bounds##*:}"
  if [[ -z "$min_mask_rate" || -z "$max_mask_rate" || "$bounds" != *:* ]]; then
    log "Invalid MASK_BOUNDS_LIST entry: $bounds"
    exit 1
  fi
  RUN_ID="week3_p4_bounds_steps${BEST_STEP_NUM_DIFFUSION_STEPS}_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_param${BASE_PARAMETERIZATION}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_min$(float_tag "$min_mask_rate")_max$(float_tag "$max_mask_rate")_${BATCH_ID}"
  if ! validate_elbo_recipe "$BEST_STEP_NUM_DIFFUSION_STEPS" "$min_mask_rate" "$max_mask_rate" >>"$RUNNER_LOG" 2>&1; then
    log "Skipping invalid ELBO recipe for $RUN_ID diffusion_steps=$BEST_STEP_NUM_DIFFUSION_STEPS min_mask_rate=$min_mask_rate max_mask_rate=$max_mask_rate"
    append_summary_row \
      "stage_e_p4_bounds" "$RUN_ID" "invalid" "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" "$BEST_STEP_NUM_DIFFUSION_STEPS" "$min_mask_rate" "$max_mask_rate" "$SCREEN_ITERATIONS" \
      "$OUT_DIR/${RUN_ID}_diffusion_last_mlx.npz" "$OUT_DIR/${RUN_ID}_diffusion_best_mlx.npz" "$OUT_DIR/${RUN_ID}_diffusion.txt" "$OUT_DIR/${RUN_ID}_diffusion_manifest.json"
    continue
  fi
  run_train "stage_e_p4_bounds" "$RUN_ID" "$BEST_STEP_NUM_DIFFUSION_STEPS" "$min_mask_rate" "$max_mask_rate" "$SCREEN_ITERATIONS" || FAILED=1
done

if [[ "$FAILED" != "0" ]]; then
  log "Batch completed with failures during the mask-bound sweep. Summary: $SUMMARY_TSV"
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: skipping metric-based winner selection, promotion check, long rerun, and full eval"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

BEST_OVERALL_INFO=()
while IFS= read -r line; do
  BEST_OVERALL_INFO+=("$line")
done < <(best_run_for_phase "stage_e_p4" 2>>"$RUNNER_LOG")

if [[ "${#BEST_OVERALL_INFO[@]}" -lt 5 ]]; then
  log "Could not determine the best overall P4 run"
  exit 1
fi

BEST_OVERALL_RUN_ID="${BEST_OVERALL_INFO[0]}"
BEST_OVERALL_NUM_DIFFUSION_STEPS="${BEST_OVERALL_INFO[1]}"
BEST_OVERALL_MIN_MASK_RATE="${BEST_OVERALL_INFO[2]}"
BEST_OVERALL_MAX_MASK_RATE="${BEST_OVERALL_INFO[3]}"
BEST_OVERALL_METRIC="${BEST_OVERALL_INFO[4]}"

log "Best overall P4 run: run_id=$BEST_OVERALL_RUN_ID diffusion_steps=$BEST_OVERALL_NUM_DIFFUSION_STEPS min_mask_rate=$BEST_OVERALL_MIN_MASK_RATE max_mask_rate=$BEST_OVERALL_MAX_MASK_RATE best_val_bpb=$BEST_OVERALL_METRIC"

PREVIOUS_SOTA_1500_METRIC="$(previous_sota_1500_metric 2>>"$RUNNER_LOG")" || {
  log "Could not determine the previous 1500-step SOTA metric"
  exit 1
}

log "Promotion gate: previous_1500_sota=$PREVIOUS_SOTA_1500_METRIC promotion_margin=$PROMOTION_MARGIN"

if ! should_promote_screen_winner "$BEST_OVERALL_METRIC" "$PREVIOUS_SOTA_1500_METRIC" "$PROMOTION_MARGIN"; then
  log "Best P4 screen run did not beat the previous 1500-step SOTA; skipping the long rerun and full eval"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

LONG_RUN_ID="${BEST_OVERALL_RUN_ID}_long${LONG_ITERATIONS}"
log "Best P4 screen run beat the previous 1500-step SOTA; promoting run_id=$BEST_OVERALL_RUN_ID to long rerun run_id=$LONG_RUN_ID"

run_train \
  "stage_e_p4_long" \
  "$LONG_RUN_ID" \
  "$BEST_OVERALL_NUM_DIFFUSION_STEPS" \
  "$BEST_OVERALL_MIN_MASK_RATE" \
  "$BEST_OVERALL_MAX_MASK_RATE" \
  "$LONG_ITERATIONS" || {
  log "Batch completed with failure during promoted P4 long rerun. Summary: $SUMMARY_TSV"
  exit 1
}

if [[ "$RUN_FULL_EVAL" != "1" ]]; then
  log "Skipping full eval because RUN_FULL_EVAL=$RUN_FULL_EVAL"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

BEST_OVERALL_CHECKPOINT="$(best_checkpoint_for_run "$LONG_RUN_ID" 2>>"$RUNNER_LOG")" || {
  log "Could not determine the best checkpoint for $LONG_RUN_ID"
  exit 1
}

run_full_eval \
  "stage_e_p4_winner" \
  "$LONG_RUN_ID" \
  "$BEST_OVERALL_CHECKPOINT" \
  "$BEST_OVERALL_NUM_DIFFUSION_STEPS" \
  "$BEST_OVERALL_MIN_MASK_RATE" \
  "$BEST_OVERALL_MAX_MASK_RATE" || exit 1

log "Batch completed successfully. Summary: $SUMMARY_TSV"
exit 0
