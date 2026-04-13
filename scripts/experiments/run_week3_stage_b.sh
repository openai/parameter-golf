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
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_b_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"
LONG_RUN_ITERATIONS="${LONG_RUN_ITERATIONS:-3000}"

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

detect_latest_stage_a_dir() {
  ls -1dt "$ROOT_DIR"/logs/week3_stage_a_* 2>/dev/null | head -1
}

STAGE_A_DIR="${STAGE_A_DIR:-$(detect_latest_stage_a_dir)}"
if [[ -z "$STAGE_A_DIR" || ! -d "$STAGE_A_DIR" ]]; then
  log "Could not find a Stage A batch directory. Set STAGE_A_DIR=/absolute/path and retry."
  exit 1
fi

WINNER_INFO=()
while IFS= read -r line; do
  WINNER_INFO+=("$line")
done < <(
  STAGE_A_DIR="$STAGE_A_DIR" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import os

stage_a_dir = Path(os.environ["STAGE_A_DIR"])
best = None
for manifest_path in sorted(stage_a_dir.glob("*_diffusion_manifest.json")):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = payload.get("run_id", "")
    if "stageA_" not in run_id:
        continue
    metric = payload.get("best_metric_value")
    checkpoint = payload.get("best_checkpoint")
    if metric is None or checkpoint is None:
        continue
    candidate = (float(metric), run_id, checkpoint)
    if best is None or candidate[0] < best[0]:
        best = candidate

if best is None:
    raise SystemExit("No Stage A winner found in manifests")

run_id = best[1]
parts = run_id.split("_")
schedule = parts[2]
timestep_sampling = parts[3]
print(run_id)
print(schedule)
print(timestep_sampling)
print(best[2])
print(f"{best[0]:.10f}")
PY
)

if [[ "${#WINNER_INFO[@]}" -lt 5 ]]; then
  log "Failed to parse Stage A winner from $STAGE_A_DIR"
  exit 1
fi

STAGE_A_WINNER_RUN_ID="${WINNER_INFO[0]}"
STAGE_A_WINNER_SCHEDULE="${WINNER_INFO[1]}"
STAGE_A_WINNER_TIMESTEP_SAMPLING="${WINNER_INFO[2]}"
STAGE_A_WINNER_CHECKPOINT="${WINNER_INFO[3]}"
STAGE_A_WINNER_METRIC="${WINNER_INFO[4]}"

WEEK3_CONFIG="$ROOT_DIR/configs/diffusion_local.env"

run_train() {
  local phase="$1"
  local run_id="$2"
  local config_path="$3"
  local mask_schedule="$4"
  local timestep_sampling="$5"
  local parameterization="$6"
  local self_conditioning="$7"
  local loss_reweighting="$8"
  local iterations_override="${9:-}"

  local train_log_path="$OUT_DIR/${run_id}_diffusion.txt"
  local last_ckpt="$OUT_DIR/${run_id}_diffusion_last_mlx.npz"
  local best_ckpt="$OUT_DIR/${run_id}_diffusion_best_mlx.npz"

  log "Starting $phase run_id=$run_id schedule=$mask_schedule timestep_sampling=$timestep_sampling self_conditioning=$self_conditioning loss_reweighting=$loss_reweighting iterations=${iterations_override:-from-config}"

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
    if [[ -n "$iterations_override" ]]; then
      export ITERATIONS="$iterations_override"
    fi
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

detect_best_stage_b_recipe() {
  SUMMARY_TSV="$SUMMARY_TSV" OUT_DIR="$OUT_DIR" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import json
import os

summary_tsv = Path(os.environ["SUMMARY_TSV"])
out_dir = Path(os.environ["OUT_DIR"])
best = None

with summary_tsv.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("phase") != "stage_b":
            continue
        if row.get("status") != "ok":
            continue
        run_id = row["run_id"]
        manifest_path = out_dir / f"{run_id}_diffusion_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metric = payload.get("best_metric_value")
        if metric is None:
            continue
        candidate = (
            float(metric),
            row["mask_schedule"],
            row["train_timestep_sampling"],
            row["parameterization"],
            row["self_conditioning"],
            row["loss_reweighting"],
            run_id,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

if best is None:
    raise SystemExit("No successful Stage B run with a saved manifest was found")

for value in best[1:]:
    print(value)
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

collect_best_stage_b_recipe() {
  local attempts="${1:-5}"
  local delay_seconds="${2:-2}"
  local attempt
  local output
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if output="$(detect_best_stage_b_recipe 2>>"$RUNNER_LOG")"; then
      printf "%s\n" "$output"
      return 0
    fi
    log "Best Stage B recipe detection attempt $attempt/$attempts failed; retrying in ${delay_seconds}s"
    sleep "$delay_seconds"
  done
  return 1
}

run_full_eval() {
  local phase="$1"
  local run_id="$2"
  local config_path="$3"
  local checkpoint_path="$4"
  local mask_schedule="$5"
  local timestep_sampling="$6"
  local parameterization="$7"
  local self_conditioning="$8"
  local loss_reweighting="$9"

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
    export MASK_SCHEDULE="$mask_schedule"
    export TRAIN_TIMESTEP_SAMPLING="$timestep_sampling"
    export PARAMETERIZATION="$parameterization"
    export SELF_CONDITIONING="$self_conditioning"
    export LOSS_REWEIGHTING="$loss_reweighting"
    "$PYTHON_BIN" diffusion_eval.py --checkpoint "$checkpoint_path"
  ) >>"$RUNNER_LOG" 2>&1; then
    log "Completed $phase full eval for run_id=$run_id"
    return 0
  fi

  log "FAILED $phase full eval for run_id=$run_id"
  return 1
}

FAILED=0

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"
log "Using Stage A batch: $STAGE_A_DIR"
log "Stage A winner: run_id=$STAGE_A_WINNER_RUN_ID schedule=$STAGE_A_WINNER_SCHEDULE timestep_sampling=$STAGE_A_WINNER_TIMESTEP_SAMPLING best_subset_val_bpb=$STAGE_A_WINNER_METRIC"

if [[ "$DRY_RUN" == "1" || -f "$STAGE_A_WINNER_CHECKPOINT" ]]; then
  run_full_eval \
    "stage_a_winner" \
    "$STAGE_A_WINNER_RUN_ID" \
    "$WEEK3_CONFIG" \
    "$STAGE_A_WINNER_CHECKPOINT" \
    "$STAGE_A_WINNER_SCHEDULE" \
    "$STAGE_A_WINNER_TIMESTEP_SAMPLING" \
    "x0" \
    "0" \
    "none" || FAILED=1
else
  log "Missing Stage A winner checkpoint: $STAGE_A_WINNER_CHECKPOINT"
  FAILED=1
fi

declare -a SELF_CONDITIONINGS=("0" "1")
declare -a LOSS_REWEIGHTINGS=("none" "inverse_mask_rate")

for self_conditioning in "${SELF_CONDITIONINGS[@]}"; do
  for loss_reweighting in "${LOSS_REWEIGHTINGS[@]}"; do
    run_id="week3_stageB_${STAGE_A_WINNER_SCHEDULE}_${STAGE_A_WINNER_TIMESTEP_SAMPLING}_sc${self_conditioning}_lw${loss_reweighting}_${BATCH_ID}"
    run_train \
      "stage_b" \
      "$run_id" \
      "$WEEK3_CONFIG" \
      "$STAGE_A_WINNER_SCHEDULE" \
      "$STAGE_A_WINNER_TIMESTEP_SAMPLING" \
      "x0" \
      "$self_conditioning" \
      "$loss_reweighting" || FAILED=1
  done
done

LONG_RUN_INFO=()
if [[ "$DRY_RUN" == "1" ]]; then
  LONG_RUN_INFO=(
    "$STAGE_A_WINNER_SCHEDULE"
    "$STAGE_A_WINNER_TIMESTEP_SAMPLING"
    "x0"
    "0"
    "none"
    "$STAGE_A_WINNER_RUN_ID"
  )
else
  LONG_RUN_INFO_RAW=""
  if LONG_RUN_INFO_RAW="$(collect_best_stage_b_recipe)"; then
    while IFS= read -r line; do
      [[ -n "$line" ]] && LONG_RUN_INFO+=("$line")
    done <<< "$LONG_RUN_INFO_RAW"
  else
    log "Could not determine the best Stage B recipe for the P2 long run"
    FAILED=1
  fi
fi

if [[ "${#LONG_RUN_INFO[@]}" -ge 6 ]]; then
  LONG_RUN_SCHEDULE="${LONG_RUN_INFO[0]}"
  LONG_RUN_TIMESTEP_SAMPLING="${LONG_RUN_INFO[1]}"
  LONG_RUN_PARAMETERIZATION="${LONG_RUN_INFO[2]}"
  LONG_RUN_SELF_CONDITIONING="${LONG_RUN_INFO[3]}"
  LONG_RUN_LOSS_REWEIGHTING="${LONG_RUN_INFO[4]}"
  LONG_RUN_SOURCE_RUN_ID="${LONG_RUN_INFO[5]}"
  LONG_RUN_RUN_ID="week3_stageB_long_${LONG_RUN_SCHEDULE}_${LONG_RUN_TIMESTEP_SAMPLING}_param${LONG_RUN_PARAMETERIZATION}_sc${LONG_RUN_SELF_CONDITIONING}_lw${LONG_RUN_LOSS_REWEIGHTING}_${BATCH_ID}"

  log "P2 long run will use source_run_id=$LONG_RUN_SOURCE_RUN_ID iterations=$LONG_RUN_ITERATIONS"
  run_train \
    "stage_b_p2" \
    "$LONG_RUN_RUN_ID" \
    "$WEEK3_CONFIG" \
    "$LONG_RUN_SCHEDULE" \
    "$LONG_RUN_TIMESTEP_SAMPLING" \
    "$LONG_RUN_PARAMETERIZATION" \
    "$LONG_RUN_SELF_CONDITIONING" \
    "$LONG_RUN_LOSS_REWEIGHTING" \
    "$LONG_RUN_ITERATIONS" || FAILED=1

  if [[ "$DRY_RUN" == "1" ]]; then
    run_full_eval \
      "stage_b_p2_winner" \
      "$LONG_RUN_RUN_ID" \
      "$WEEK3_CONFIG" \
      "$OUT_DIR/${LONG_RUN_RUN_ID}_diffusion_best_mlx.npz" \
      "$LONG_RUN_SCHEDULE" \
      "$LONG_RUN_TIMESTEP_SAMPLING" \
      "$LONG_RUN_PARAMETERIZATION" \
      "$LONG_RUN_SELF_CONDITIONING" \
      "$LONG_RUN_LOSS_REWEIGHTING" || FAILED=1
  else
    if LONG_RUN_CHECKPOINT="$(best_checkpoint_for_run "$LONG_RUN_RUN_ID")"; then
      run_full_eval \
        "stage_b_p2_winner" \
        "$LONG_RUN_RUN_ID" \
        "$WEEK3_CONFIG" \
        "$LONG_RUN_CHECKPOINT" \
        "$LONG_RUN_SCHEDULE" \
        "$LONG_RUN_TIMESTEP_SAMPLING" \
        "$LONG_RUN_PARAMETERIZATION" \
        "$LONG_RUN_SELF_CONDITIONING" \
        "$LONG_RUN_LOSS_REWEIGHTING" || FAILED=1
    else
      log "Could not determine the checkpoint for the P2 long run winner: $LONG_RUN_RUN_ID"
      FAILED=1
    fi
  fi
fi

if [[ "$FAILED" == "0" ]]; then
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

log "Batch completed with failures. Summary: $SUMMARY_TSV"
exit 1
