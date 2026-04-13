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
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_c_length_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/diffusion_local.env}"
LONG_RUN_ITERATIONS="${LONG_RUN_ITERATIONS:-3000}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"

# Current promoted week-3 recipe. Override any of these via env vars if needed.
MASK_SCHEDULE="${MASK_SCHEDULE:-linear}"
TRAIN_TIMESTEP_SAMPLING="${TRAIN_TIMESTEP_SAMPLING:-cyclic}"
PARAMETERIZATION="${PARAMETERIZATION:-x0}"
SELF_CONDITIONING="${SELF_CONDITIONING:-0}"
LOSS_REWEIGHTING="${LOSS_REWEIGHTING:-none}"

RUN_ID="${RUN_ID:-week3_p2_long_${MASK_SCHEDULE}_${TRAIN_TIMESTEP_SAMPLING}_param${PARAMETERIZATION}_sc${SELF_CONDITIONING}_lw${LOSS_REWEIGHTING}_${BATCH_ID}}"

REQUESTED_RUN_ID="$RUN_ID"
REQUESTED_MASK_SCHEDULE="$MASK_SCHEDULE"
REQUESTED_TRAIN_TIMESTEP_SAMPLING="$TRAIN_TIMESTEP_SAMPLING"
REQUESTED_PARAMETERIZATION="$PARAMETERIZATION"
REQUESTED_SELF_CONDITIONING="$SELF_CONDITIONING"
REQUESTED_LOSS_REWEIGHTING="$LOSS_REWEIGHTING"

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

TRAIN_LOG_PATH="$OUT_DIR/${RUN_ID}_diffusion.txt"
LAST_CKPT="$OUT_DIR/${RUN_ID}_diffusion_last_mlx.npz"
BEST_CKPT="$OUT_DIR/${RUN_ID}_diffusion_best_mlx.npz"

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"
log "Running P2 longer local experiment with run_id=$REQUESTED_RUN_ID"
log "Recipe: schedule=$REQUESTED_MASK_SCHEDULE timestep_sampling=$REQUESTED_TRAIN_TIMESTEP_SAMPLING parameterization=$REQUESTED_PARAMETERIZATION self_conditioning=$REQUESTED_SELF_CONDITIONING loss_reweighting=$REQUESTED_LOSS_REWEIGHTING iterations=$LONG_RUN_ITERATIONS"

if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN: skipping train_diffusion.py"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "stage_c_p2" "$REQUESTED_RUN_ID" "dry-run" "$(basename "$CONFIG_PATH")" "$REQUESTED_MASK_SCHEDULE" "$REQUESTED_TRAIN_TIMESTEP_SAMPLING" \
    "$REQUESTED_PARAMETERIZATION" "$REQUESTED_SELF_CONDITIONING" "$REQUESTED_LOSS_REWEIGHTING" "$LONG_RUN_ITERATIONS" \
    "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH" >> "$SUMMARY_TSV"
else
  if (
    set -a
    source "$CONFIG_PATH"
    set +a
    export RUN_ID="$REQUESTED_RUN_ID"
    export OUT_DIR="$OUT_DIR"
    export ITERATIONS="$LONG_RUN_ITERATIONS"
    export MASK_SCHEDULE="$REQUESTED_MASK_SCHEDULE"
    export TRAIN_TIMESTEP_SAMPLING="$REQUESTED_TRAIN_TIMESTEP_SAMPLING"
    export PARAMETERIZATION="$REQUESTED_PARAMETERIZATION"
    export SELF_CONDITIONING="$REQUESTED_SELF_CONDITIONING"
    export LOSS_REWEIGHTING="$REQUESTED_LOSS_REWEIGHTING"
    "$PYTHON_BIN" train_diffusion.py
  ) >>"$RUNNER_LOG" 2>&1; then
    log "Completed stage_c_p2 run_id=$REQUESTED_RUN_ID"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "stage_c_p2" "$REQUESTED_RUN_ID" "ok" "$(basename "$CONFIG_PATH")" "$REQUESTED_MASK_SCHEDULE" "$REQUESTED_TRAIN_TIMESTEP_SAMPLING" \
      "$REQUESTED_PARAMETERIZATION" "$REQUESTED_SELF_CONDITIONING" "$REQUESTED_LOSS_REWEIGHTING" "$LONG_RUN_ITERATIONS" \
      "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH" >> "$SUMMARY_TSV"
  else
    log "FAILED stage_c_p2 run_id=$REQUESTED_RUN_ID"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "stage_c_p2" "$REQUESTED_RUN_ID" "failed" "$(basename "$CONFIG_PATH")" "$REQUESTED_MASK_SCHEDULE" "$REQUESTED_TRAIN_TIMESTEP_SAMPLING" \
      "$REQUESTED_PARAMETERIZATION" "$REQUESTED_SELF_CONDITIONING" "$REQUESTED_LOSS_REWEIGHTING" "$LONG_RUN_ITERATIONS" \
      "$LAST_CKPT" "$BEST_CKPT" "$TRAIN_LOG_PATH" >> "$SUMMARY_TSV"
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

if ! BEST_CHECKPOINT_PATH="$(best_checkpoint_for_run "$REQUESTED_RUN_ID" 2>>"$RUNNER_LOG")"; then
  log "Could not determine the checkpoint to full-eval for run_id=$REQUESTED_RUN_ID"
  exit 1
fi

log "Starting stage_c_p2 full eval for run_id=$REQUESTED_RUN_ID checkpoint=$(basename "$BEST_CHECKPOINT_PATH")"
if (
  set -a
  source "$CONFIG_PATH"
  set +a
  export OUT_DIR="$OUT_DIR"
  export VAL_MAX_TOKENS=0
  export MASK_SCHEDULE="$REQUESTED_MASK_SCHEDULE"
  export TRAIN_TIMESTEP_SAMPLING="$REQUESTED_TRAIN_TIMESTEP_SAMPLING"
  export PARAMETERIZATION="$REQUESTED_PARAMETERIZATION"
  export SELF_CONDITIONING="$REQUESTED_SELF_CONDITIONING"
  export LOSS_REWEIGHTING="$REQUESTED_LOSS_REWEIGHTING"
  "$PYTHON_BIN" diffusion_eval.py --checkpoint "$BEST_CHECKPOINT_PATH"
) >>"$RUNNER_LOG" 2>&1; then
  log "Completed stage_c_p2 full eval for run_id=$REQUESTED_RUN_ID"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

log "FAILED stage_c_p2 full eval for run_id=$REQUESTED_RUN_ID"
exit 1
