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
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/week3_stage_f_optim_${BATCH_ID}}"
RUNNER_LOG="$OUT_DIR/runner.log"
SUMMARY_TSV="$OUT_DIR/summary.tsv"
SEARCH_STATE_PATH="$OUT_DIR/search_state.json"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/diffusion_local.env}"
SCREEN_ITERATIONS="${SCREEN_ITERATIONS:-1500}"
LONG_ITERATIONS="${LONG_ITERATIONS:-3000}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"

SEARCH_PARAM_ORDER="${SEARCH_PARAM_ORDER:-grad_clip_norm,warmup_steps,beta2,learning_rate}"
IMPROVEMENT_THRESHOLD="${IMPROVEMENT_THRESHOLD:-0.003}"
WORSE_GUARD_THRESHOLD="${WORSE_GUARD_THRESHOLD:-0.03}"
CURRENT_FULL_VAL_BPB="${CURRENT_FULL_VAL_BPB:-2.5005}"
MAX_SEARCH_PASSES="${MAX_SEARCH_PASSES:-10}"

GRAD_CLIP_INITIAL_DELTA="${GRAD_CLIP_INITIAL_DELTA:-0.1}"
GRAD_CLIP_MIN_VALUE="${GRAD_CLIP_MIN_VALUE:-0.05}"
WARMUP_INITIAL_DELTA="${WARMUP_INITIAL_DELTA:-20}"
WARMUP_MAX_VALUE="${WARMUP_MAX_VALUE:-160}"
BETA2_INITIAL_DELTA="${BETA2_INITIAL_DELTA:-0.03}"
BETA2_MIN_VALUE="${BETA2_MIN_VALUE:-0.80}"
LEARNING_RATE_INITIAL_DELTA="${LEARNING_RATE_INITIAL_DELTA:-0.0001}"
LEARNING_RATE_MAX_VALUE="${LEARNING_RATE_MAX_VALUE:-0.0012}"

mkdir -p "$OUT_DIR"
touch "$RUNNER_LOG"

printf "phase\trun_id\tstatus\tpass_num\tsearch_param\tattempt_index\tdelta_value\tcandidate_value\tdecision\tdecision_reason\tincumbent_before_metric\tincumbent_after_metric\tcandidate_metric\timprovement\tfull_eval_bpb\tmask_schedule\ttrain_timestep_sampling\tparameterization\tself_conditioning\tloss_reweighting\tnum_diffusion_steps\tmin_mask_rate\tmax_mask_rate\tlearning_rate\tweight_decay\tbeta2\tgrad_clip_norm\twarmup_steps\titerations\tcheckpoint\tbest_checkpoint\tlog_path\n" > "$SUMMARY_TSV"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  local msg="$1"
  printf "[%s] %s\n" "$(timestamp)" "$msg" | tee -a "$RUNNER_LOG"
}

append_summary_row() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" \
    "$BASE_MASK_SCHEDULE" "$BASE_TRAIN_TIMESTEP_SAMPLING" "$BASE_PARAMETERIZATION" "$BASE_SELF_CONDITIONING" "$BASE_LOSS_REWEIGHTING" \
    "$BASE_NUM_DIFFUSION_STEPS" "$BASE_MIN_MASK_RATE" "$BASE_MAX_MASK_RATE" "${16}" "${17}" "${18}" "${19}" "${20}" "${21}" "${22}" "${23}" "${24}" >> "$SUMMARY_TSV"
}

float_tag() {
  local value
  value="$(normalize_number "$1")"
  value="${value//-/m}"
  value="${value//./p}"
  echo "$value"
}

normalize_number() {
  VALUE="$1" "$PYTHON_BIN" - <<'PY'
import math
import os

value = float(os.environ["VALUE"])
if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
    print(int(round(value)))
else:
    print(format(value, ".12g"))
PY
}

float_eq() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import math
import os
import sys

a = float(os.environ["A"])
b = float(os.environ["B"])
sys.exit(0 if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) else 1)
PY
}

float_add() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import os
value = float(os.environ["A"]) + float(os.environ["B"])
print(format(value, ".12g"))
PY
}

float_sub() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import os
value = float(os.environ["A"]) - float(os.environ["B"])
print(format(value, ".12g"))
PY
}

float_mul() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import os
value = float(os.environ["A"]) * float(os.environ["B"])
print(format(value, ".12g"))
PY
}

float_ge() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import os
import sys

a = float(os.environ["A"])
b = float(os.environ["B"])
sys.exit(0 if a >= b else 1)
PY
}

float_le() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import os
import sys

a = float(os.environ["A"])
b = float(os.environ["B"])
sys.exit(0 if a <= b else 1)
PY
}

int_add() {
  A="$1" B="$2" "$PYTHON_BIN" - <<'PY'
import os
print(int(float(os.environ["A"])) + int(float(os.environ["B"])))
PY
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
  BASE_LEARNING_RATE="${LEARNING_RATE}"
  BASE_WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
  BASE_BETA2="${BETA2:-0.95}"
  BASE_GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
  BASE_WARMUP_STEPS="${WARMUP_STEPS:-5}"
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
  local expected_learning_rate="${11}"
  local expected_weight_decay="${12}"
  local expected_beta2="${13}"
  local expected_grad_clip_norm="${14}"
  local expected_warmup_steps="${15}"

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
  EXPECTED_LEARNING_RATE="$expected_learning_rate" \
  EXPECTED_WEIGHT_DECAY="$expected_weight_decay" \
  EXPECTED_BETA2="$expected_beta2" \
  EXPECTED_GRAD_CLIP_NORM="$expected_grad_clip_norm" \
  EXPECTED_WARMUP_STEPS="$expected_warmup_steps" \
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
):
    actual_value = float(payload.get(key))
    expected_value = float(os.environ[env_name])
    if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            f"Recipe mismatch for {key}: expected {expected_value!r}, got {actual_value!r}"
        )
PY
}

build_run_id() {
  local tag="$1"
  local lr="$2"
  local wd="$3"
  local beta2="$4"
  local clip="$5"
  local warmup="$6"

  echo "week3_p5_${tag}_${BASE_MASK_SCHEDULE}_${BASE_TRAIN_TIMESTEP_SAMPLING}_param${BASE_PARAMETERIZATION}_sc${BASE_SELF_CONDITIONING}_lw${BASE_LOSS_REWEIGHTING}_steps${BASE_NUM_DIFFUSION_STEPS}_min$(float_tag "$BASE_MIN_MASK_RATE")_max$(float_tag "$BASE_MAX_MASK_RATE")_lr$(float_tag "$lr")_wd$(float_tag "$wd")_b2$(float_tag "$beta2")_clip$(float_tag "$clip")_wu${warmup}_${BATCH_ID}"
}

run_train() {
  local phase="$1"
  local run_id="$2"
  local learning_rate="$3"
  local weight_decay="$4"
  local beta2="$5"
  local grad_clip_norm="$6"
  local warmup_steps="$7"
  local iterations="$8"

  LAST_RUN_PHASE="$phase"
  LAST_RUN_ID="$run_id"
  LAST_RUN_LOG_PATH="$OUT_DIR/${run_id}_diffusion.txt"
  LAST_RUN_LAST_CKPT="$OUT_DIR/${run_id}_diffusion_last_mlx.npz"
  LAST_RUN_BEST_CKPT="$OUT_DIR/${run_id}_diffusion_best_mlx.npz"
  LAST_RUN_STATUS="failed"

  log "Starting $phase run_id=$run_id schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING parameterization=$BASE_PARAMETERIZATION self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING diffusion_steps=$BASE_NUM_DIFFUSION_STEPS min_mask_rate=$BASE_MIN_MASK_RATE max_mask_rate=$BASE_MAX_MASK_RATE lr=$learning_rate wd=$weight_decay beta2=$beta2 grad_clip_norm=$grad_clip_norm warmup_steps=$warmup_steps iterations=$iterations"

  if [[ "$DRY_RUN" == "1" ]]; then
    LAST_RUN_STATUS="dry-run"
    log "DRY RUN: skipping train_diffusion.py for $run_id"
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
    export NUM_DIFFUSION_STEPS="$BASE_NUM_DIFFUSION_STEPS"
    export MIN_MASK_RATE="$BASE_MIN_MASK_RATE"
    export MAX_MASK_RATE="$BASE_MAX_MASK_RATE"
    export LEARNING_RATE="$learning_rate"
    export WEIGHT_DECAY="$weight_decay"
    export BETA2="$beta2"
    export GRAD_CLIP_NORM="$grad_clip_norm"
    export WARMUP_STEPS="$warmup_steps"
    "$PYTHON_BIN" train_diffusion.py
  ) >>"$RUNNER_LOG" 2>&1; then
    if ! assert_recipe_matches_log \
      "$LAST_RUN_LOG_PATH" \
      "$run_id" \
      "$BASE_MASK_SCHEDULE" \
      "$BASE_TRAIN_TIMESTEP_SAMPLING" \
      "$BASE_PARAMETERIZATION" \
      "$BASE_SELF_CONDITIONING" \
      "$BASE_LOSS_REWEIGHTING" \
      "$BASE_NUM_DIFFUSION_STEPS" \
      "$BASE_MIN_MASK_RATE" \
      "$BASE_MAX_MASK_RATE" \
      "$learning_rate" \
      "$weight_decay" \
      "$beta2" \
      "$grad_clip_norm" \
      "$warmup_steps" >>"$RUNNER_LOG" 2>&1; then
      LAST_RUN_STATUS="failed"
      log "FAILED $phase run_id=$run_id due to recipe verification mismatch"
      return 1
    fi
    LAST_RUN_STATUS="ok"
    log "Completed $phase run_id=$run_id"
    return 0
  fi

  log "FAILED $phase run_id=$run_id"
  LAST_RUN_STATUS="failed"
  return 1
}

run_full_eval() {
  local phase="$1"
  local run_id="$2"
  local checkpoint_path="$3"
  local learning_rate="$4"
  local weight_decay="$5"
  local beta2="$6"
  local grad_clip_norm="$7"
  local warmup_steps="$8"

  LAST_FULL_EVAL_STATUS="failed"
  LAST_FULL_EVAL_LOG_PATH="${checkpoint_path%.npz}_full_eval.txt"

  log "Starting $phase full eval for run_id=$run_id checkpoint=$(basename "$checkpoint_path") lr=$learning_rate wd=$weight_decay beta2=$beta2 grad_clip_norm=$grad_clip_norm warmup_steps=$warmup_steps"

  if [[ "$DRY_RUN" == "1" ]]; then
    LAST_FULL_EVAL_STATUS="dry-run"
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
    export PARAMETERIZATION="$BASE_PARAMETERIZATION"
    export SELF_CONDITIONING="$BASE_SELF_CONDITIONING"
    export LOSS_REWEIGHTING="$BASE_LOSS_REWEIGHTING"
    export NUM_DIFFUSION_STEPS="$BASE_NUM_DIFFUSION_STEPS"
    export MIN_MASK_RATE="$BASE_MIN_MASK_RATE"
    export MAX_MASK_RATE="$BASE_MAX_MASK_RATE"
    export LEARNING_RATE="$learning_rate"
    export WEIGHT_DECAY="$weight_decay"
    export BETA2="$beta2"
    export GRAD_CLIP_NORM="$grad_clip_norm"
    export WARMUP_STEPS="$warmup_steps"
    "$PYTHON_BIN" diffusion_eval.py --checkpoint "$checkpoint_path"
  ) >>"$RUNNER_LOG" 2>&1; then
    LAST_FULL_EVAL_STATUS="ok"
    log "Completed $phase full eval for run_id=$run_id"
    return 0
  fi

  log "FAILED $phase full eval for run_id=$run_id"
  LAST_FULL_EVAL_STATUS="failed"
  return 1
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

dry_run_metric() {
  local learning_rate="$1"
  local weight_decay="$2"
  local beta2="$3"
  local grad_clip_norm="$4"
  local warmup_steps="$5"
  local iterations="$6"

  LEARNING_RATE="$learning_rate" \
  WEIGHT_DECAY="$weight_decay" \
  BETA2="$beta2" \
  GRAD_CLIP_NORM="$grad_clip_norm" \
  WARMUP_STEPS="$warmup_steps" \
  ITERATIONS="$iterations" \
  "$PYTHON_BIN" - <<'PY'
import os

lr = float(os.environ["LEARNING_RATE"])
wd = float(os.environ["WEIGHT_DECAY"])
beta2 = float(os.environ["BETA2"])
clip = float(os.environ["GRAD_CLIP_NORM"])
warmup = float(os.environ["WARMUP_STEPS"])
iterations = int(float(os.environ["ITERATIONS"]))

# Synthetic dry-run objective so the search loop exercises both accept and reject paths.
metric = 2.8100
metric += 800000.0 * (lr - 0.0005) ** 2
metric += 5.0 * (beta2 - 0.92) ** 2
metric += 0.50 * (clip - 0.18) ** 2
metric += 0.000008 * (warmup - 45.0) ** 2
metric += 50.0 * wd
if iterations > 1500:
    metric -= 0.3200
print(f"{metric:.10f}")
PY
}

metric_info_for_run() {
  local run_id="$1"
  local learning_rate="$2"
  local weight_decay="$3"
  local beta2="$4"
  local grad_clip_norm="$5"
  local warmup_steps="$6"
  local iterations="$7"

  if [[ "$DRY_RUN" == "1" ]]; then
    local metric
    metric="$(dry_run_metric "$learning_rate" "$weight_decay" "$beta2" "$grad_clip_norm" "$warmup_steps" "$iterations")" || return 1
    printf "%s\n%s\n" "$metric" "$iterations"
    return 0
  fi

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
best_step = payload.get("best_step")
if metric is None:
    raise SystemExit(f"No best_metric_value recorded for {run_id}")
print(metric)
print(best_step if best_step is not None else "")
PY
}

log_has_nan() {
  local log_path="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 1
  fi

  LOG_PATH="$log_path" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import re
import sys

log_path = Path(os.environ["LOG_PATH"])
if not log_path.exists():
    sys.exit(1)
text = log_path.read_bytes().decode("utf-8", errors="ignore")
pattern = re.compile(r"(^|[^a-z])nan([^a-z]|$)", re.IGNORECASE)
sys.exit(0 if pattern.search(text) else 1)
PY
}

full_eval_bpb_from_log() {
  local eval_log_path="$1"
  EVAL_LOG_PATH="$eval_log_path" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import re

path = Path(os.environ["EVAL_LOG_PATH"])
text = path.read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"final_diffusion_eval .*?val_bpb:([0-9.]+)", text)
if not matches:
    raise SystemExit(f"No final_diffusion_eval val_bpb found in {path}")
print(matches[-1])
PY
}

candidate_value_for_param() {
  local param="$1"
  local current_value="$2"
  local delta="$3"

  PARAM="$param" CURRENT_VALUE="$current_value" DELTA="$delta" \
  GRAD_CLIP_MIN_VALUE="$GRAD_CLIP_MIN_VALUE" \
  WARMUP_MAX_VALUE="$WARMUP_MAX_VALUE" \
  BETA2_MIN_VALUE="$BETA2_MIN_VALUE" \
  LEARNING_RATE_MAX_VALUE="$LEARNING_RATE_MAX_VALUE" \
  "$PYTHON_BIN" - <<'PY'
import os

param = os.environ["PARAM"]
current = float(os.environ["CURRENT_VALUE"])
delta = float(os.environ["DELTA"])

if param == "grad_clip_norm":
    candidate = max(float(os.environ["GRAD_CLIP_MIN_VALUE"]), current - delta)
elif param == "warmup_steps":
    candidate = min(float(os.environ["WARMUP_MAX_VALUE"]), current + delta)
elif param == "beta2":
    candidate = max(float(os.environ["BETA2_MIN_VALUE"]), current - delta)
elif param == "learning_rate":
    candidate = min(float(os.environ["LEARNING_RATE_MAX_VALUE"]), current + delta)
else:
    raise SystemExit(f"Unsupported parameter {param!r}")

if param == "warmup_steps":
    candidate = int(round(candidate))
elif abs(candidate - round(candidate)) <= 1e-12:
    candidate = int(round(candidate))

print(format(candidate, ".12g") if isinstance(candidate, float) else candidate)
PY
}

initial_delta_for_param() {
  local param="$1"
  case "$param" in
    grad_clip_norm) echo "$GRAD_CLIP_INITIAL_DELTA" ;;
    warmup_steps) echo "$WARMUP_INITIAL_DELTA" ;;
    beta2) echo "$BETA2_INITIAL_DELTA" ;;
    learning_rate) echo "$LEARNING_RATE_INITIAL_DELTA" ;;
    *)
      log "Unsupported parameter in initial_delta_for_param: $param"
      return 1
      ;;
  esac
}

current_value_for_param() {
  local param="$1"
  case "$param" in
    grad_clip_norm) echo "$CURRENT_GRAD_CLIP_NORM" ;;
    warmup_steps) echo "$CURRENT_WARMUP_STEPS" ;;
    beta2) echo "$CURRENT_BETA2" ;;
    learning_rate) echo "$CURRENT_LEARNING_RATE" ;;
    *)
      log "Unsupported parameter in current_value_for_param: $param"
      return 1
      ;;
  esac
}

apply_candidate_recipe() {
  local param="$1"
  local candidate_value="$2"

  CANDIDATE_LEARNING_RATE="$CURRENT_LEARNING_RATE"
  CANDIDATE_WEIGHT_DECAY="$CURRENT_WEIGHT_DECAY"
  CANDIDATE_BETA2="$CURRENT_BETA2"
  CANDIDATE_GRAD_CLIP_NORM="$CURRENT_GRAD_CLIP_NORM"
  CANDIDATE_WARMUP_STEPS="$CURRENT_WARMUP_STEPS"

  case "$param" in
    grad_clip_norm) CANDIDATE_GRAD_CLIP_NORM="$candidate_value" ;;
    warmup_steps) CANDIDATE_WARMUP_STEPS="$candidate_value" ;;
    beta2) CANDIDATE_BETA2="$candidate_value" ;;
    learning_rate) CANDIDATE_LEARNING_RATE="$candidate_value" ;;
    *)
      log "Unsupported parameter in apply_candidate_recipe: $param"
      return 1
      ;;
  esac
}

write_search_state() {
  SUMMARY_TSV="$SUMMARY_TSV" \
  SEARCH_STATE_PATH="$SEARCH_STATE_PATH" \
  BATCH_ID="$BATCH_ID" \
  OUT_DIR="$OUT_DIR" \
  CONFIG_PATH="$CONFIG_PATH" \
  DRY_RUN="$DRY_RUN" \
  SCREEN_ITERATIONS="$SCREEN_ITERATIONS" \
  LONG_ITERATIONS="$LONG_ITERATIONS" \
  IMPROVEMENT_THRESHOLD="$IMPROVEMENT_THRESHOLD" \
  WORSE_GUARD_THRESHOLD="$WORSE_GUARD_THRESHOLD" \
  CURRENT_FULL_VAL_BPB="$CURRENT_FULL_VAL_BPB" \
  CURRENT_RUN_ID="${CURRENT_RUN_ID:-}" \
  CURRENT_SCREEN_METRIC="${CURRENT_SCREEN_METRIC:-}" \
  CURRENT_LEARNING_RATE="${CURRENT_LEARNING_RATE:-}" \
  CURRENT_WEIGHT_DECAY="${CURRENT_WEIGHT_DECAY:-}" \
  CURRENT_BETA2="${CURRENT_BETA2:-}" \
  CURRENT_GRAD_CLIP_NORM="${CURRENT_GRAD_CLIP_NORM:-}" \
  CURRENT_WARMUP_STEPS="${CURRENT_WARMUP_STEPS:-}" \
  TOTAL_ACCEPTED_CHANGES="${TOTAL_ACCEPTED_CHANGES:-0}" \
  FINAL_LONG_RUN_ID="${FINAL_LONG_RUN_ID:-}" \
  FINAL_LONG_RUN_METRIC="${FINAL_LONG_RUN_METRIC:-}" \
  FINAL_FULL_EVAL_BPB="${FINAL_FULL_EVAL_BPB:-}" \
  FINAL_FULL_EVAL_STATUS="${FINAL_FULL_EVAL_STATUS:-}" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import csv
import json
import os

summary_path = Path(os.environ["SUMMARY_TSV"])
rows = []
if summary_path.exists():
    with summary_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

passes = []
for row in rows:
    if row.get("phase") != "stage_f_p5_search":
        continue
    pass_num = int(row["pass_num"])
    while len(passes) < pass_num:
        passes.append({"pass_num": len(passes) + 1, "attempts": []})
    passes[pass_num - 1]["attempts"].append(
        {
            "run_id": row["run_id"],
            "search_param": row["search_param"],
            "attempt_index": int(row["attempt_index"]),
            "delta_value": row["delta_value"],
            "candidate_value": row["candidate_value"],
            "status": row["status"],
            "decision": row["decision"],
            "decision_reason": row["decision_reason"],
            "incumbent_before_metric": row["incumbent_before_metric"],
            "incumbent_after_metric": row["incumbent_after_metric"],
            "candidate_metric": row["candidate_metric"],
            "improvement": row["improvement"],
        }
    )

accepted_path = [
    {
        "pass_num": int(row["pass_num"]),
        "search_param": row["search_param"],
        "run_id": row["run_id"],
        "candidate_value": row["candidate_value"],
        "candidate_metric": row["candidate_metric"],
        "improvement": row["improvement"],
    }
    for row in rows
    if row.get("phase") == "stage_f_p5_search" and row.get("decision") == "accepted"
]

state = {
    "batch_id": os.environ["BATCH_ID"],
    "out_dir": os.environ["OUT_DIR"],
    "config_path": os.environ["CONFIG_PATH"],
    "dry_run": os.environ["DRY_RUN"] == "1",
    "screen_iterations": int(os.environ["SCREEN_ITERATIONS"]),
    "long_iterations": int(os.environ["LONG_ITERATIONS"]),
    "improvement_threshold": float(os.environ["IMPROVEMENT_THRESHOLD"]),
    "worse_guard_threshold": float(os.environ["WORSE_GUARD_THRESHOLD"]),
    "current_full_val_bpb": float(os.environ["CURRENT_FULL_VAL_BPB"]),
    "total_accepted_changes": int(os.environ["TOTAL_ACCEPTED_CHANGES"]),
    "incumbent": {
        "run_id": os.environ["CURRENT_RUN_ID"],
        "screen_metric": os.environ["CURRENT_SCREEN_METRIC"],
        "learning_rate": os.environ["CURRENT_LEARNING_RATE"],
        "weight_decay": os.environ["CURRENT_WEIGHT_DECAY"],
        "beta2": os.environ["CURRENT_BETA2"],
        "grad_clip_norm": os.environ["CURRENT_GRAD_CLIP_NORM"],
        "warmup_steps": os.environ["CURRENT_WARMUP_STEPS"],
    },
    "accepted_path": accepted_path,
    "passes": passes,
    "final_long_run": {
        "run_id": os.environ["FINAL_LONG_RUN_ID"],
        "best_screen_metric": os.environ["FINAL_LONG_RUN_METRIC"],
        "full_eval_bpb": os.environ["FINAL_FULL_EVAL_BPB"],
        "full_eval_status": os.environ["FINAL_FULL_EVAL_STATUS"],
    },
}

Path(os.environ["SEARCH_STATE_PATH"]).write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

search_parameter() {
  local pass_num="$1"
  local param="$2"
  local attempt_index=1
  local delta
  delta="$(initial_delta_for_param "$param")" || return 1

  log "Starting pass=$pass_num parameter=$param from incumbent run_id=$CURRENT_RUN_ID metric=$CURRENT_SCREEN_METRIC value=$(current_value_for_param "$param") delta=$delta"

  while true; do
    local current_value
    current_value="$(current_value_for_param "$param")" || return 1

    local candidate_value
    candidate_value="$(candidate_value_for_param "$param" "$current_value" "$delta")" || return 1

    if float_eq "$candidate_value" "$current_value"; then
      log "Stopping pass=$pass_num parameter=$param because boundary has been reached at value=$current_value"
      return 0
    fi

    apply_candidate_recipe "$param" "$candidate_value" || return 1

    local run_tag="dynamic_p${pass_num}_${param}_a${attempt_index}"
    local run_id
    run_id="$(build_run_id "$run_tag" "$CANDIDATE_LEARNING_RATE" "$CANDIDATE_WEIGHT_DECAY" "$CANDIDATE_BETA2" "$CANDIDATE_GRAD_CLIP_NORM" "$CANDIDATE_WARMUP_STEPS")"

    local incumbent_before_metric="$CURRENT_SCREEN_METRIC"
    local incumbent_after_metric="$CURRENT_SCREEN_METRIC"
    local candidate_metric=""
    local improvement=""
    local decision="rejected"
    local decision_reason="train_failed"
    local row_status="failed"

    if run_train \
      "stage_f_p5_search" \
      "$run_id" \
      "$CANDIDATE_LEARNING_RATE" \
      "$CANDIDATE_WEIGHT_DECAY" \
      "$CANDIDATE_BETA2" \
      "$CANDIDATE_GRAD_CLIP_NORM" \
      "$CANDIDATE_WARMUP_STEPS" \
      "$SCREEN_ITERATIONS"; then
      row_status="$LAST_RUN_STATUS"

      if [[ "$DRY_RUN" != "1" ]] && log_has_nan "$LAST_RUN_LOG_PATH"; then
        row_status="invalid"
        decision_reason="nan_detected"
      else
        local metric_info=()
        while IFS= read -r line; do
          metric_info+=("$line")
        done < <(metric_info_for_run \
          "$run_id" \
          "$CANDIDATE_LEARNING_RATE" \
          "$CANDIDATE_WEIGHT_DECAY" \
          "$CANDIDATE_BETA2" \
          "$CANDIDATE_GRAD_CLIP_NORM" \
          "$CANDIDATE_WARMUP_STEPS" \
          "$SCREEN_ITERATIONS" 2>>"$RUNNER_LOG")

        if [[ "${#metric_info[@]}" -ge 1 && -n "${metric_info[0]}" ]]; then
          candidate_metric="${metric_info[0]}"
          improvement="$(float_sub "$incumbent_before_metric" "$candidate_metric")"

          if float_ge "$candidate_metric" "$(float_add "$incumbent_before_metric" "$WORSE_GUARD_THRESHOLD")"; then
            decision_reason="worse_than_guard"
          elif float_ge "$improvement" "$IMPROVEMENT_THRESHOLD"; then
            decision="accepted"
            decision_reason="improved"
            incumbent_after_metric="$candidate_metric"
            CURRENT_RUN_ID="$run_id"
            CURRENT_SCREEN_METRIC="$candidate_metric"
            CURRENT_LEARNING_RATE="$CANDIDATE_LEARNING_RATE"
            CURRENT_WEIGHT_DECAY="$CANDIDATE_WEIGHT_DECAY"
            CURRENT_BETA2="$CANDIDATE_BETA2"
            CURRENT_GRAD_CLIP_NORM="$CANDIDATE_GRAD_CLIP_NORM"
            CURRENT_WARMUP_STEPS="$CANDIDATE_WARMUP_STEPS"
            PASS_ACCEPTED_CHANGES=$((PASS_ACCEPTED_CHANGES + 1))
            TOTAL_ACCEPTED_CHANGES=$((TOTAL_ACCEPTED_CHANGES + 1))
          else
            decision_reason="no_material_improvement"
          fi
        else
          row_status="invalid"
          decision_reason="missing_metric"
        fi
      fi
    fi

    append_summary_row \
      "stage_f_p5_search" "$run_id" "$row_status" "$pass_num" "$param" "$attempt_index" "$delta" "$candidate_value" "$decision" "$decision_reason" \
      "$incumbent_before_metric" "$incumbent_after_metric" "$candidate_metric" "$improvement" "" \
      "$CANDIDATE_LEARNING_RATE" "$CANDIDATE_WEIGHT_DECAY" "$CANDIDATE_BETA2" "$CANDIDATE_GRAD_CLIP_NORM" "$CANDIDATE_WARMUP_STEPS" "$SCREEN_ITERATIONS" \
      "$LAST_RUN_LAST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
    write_search_state

    if [[ "$decision" == "accepted" ]]; then
      log "Accepted pass=$pass_num parameter=$param attempt=$attempt_index value=$candidate_value metric=$candidate_metric improvement=$improvement"
      delta="$(float_mul "$delta" "2.0")"
      attempt_index=$((attempt_index + 1))
      continue
    fi

    log "Stopping pass=$pass_num parameter=$param at attempt=$attempt_index reason=$decision_reason candidate_value=$candidate_value candidate_metric=${candidate_metric:-n/a}"
    return 0
  done
}

load_base_recipe_from_config

CURRENT_RUN_ID=""
CURRENT_SCREEN_METRIC=""
CURRENT_LEARNING_RATE="$BASE_LEARNING_RATE"
CURRENT_WEIGHT_DECAY="$BASE_WEIGHT_DECAY"
CURRENT_BETA2="$BASE_BETA2"
CURRENT_GRAD_CLIP_NORM="$BASE_GRAD_CLIP_NORM"
CURRENT_WARMUP_STEPS="$BASE_WARMUP_STEPS"
TOTAL_ACCEPTED_CHANGES=0
FINAL_LONG_RUN_ID=""
FINAL_LONG_RUN_METRIC=""
FINAL_FULL_EVAL_BPB=""
FINAL_FULL_EVAL_STATUS=""

log "Batch starting BATCH_ID=$BATCH_ID OUT_DIR=$OUT_DIR"
log "Base recipe source: $(basename "$CONFIG_PATH")"
log "Base recipe: run_id=${BASE_CONFIG_RUN_ID:-from-config} schedule=$BASE_MASK_SCHEDULE timestep_sampling=$BASE_TRAIN_TIMESTEP_SAMPLING parameterization=$BASE_PARAMETERIZATION self_conditioning=$BASE_SELF_CONDITIONING loss_reweighting=$BASE_LOSS_REWEIGHTING diffusion_steps=$BASE_NUM_DIFFUSION_STEPS min_mask_rate=$BASE_MIN_MASK_RATE max_mask_rate=$BASE_MAX_MASK_RATE"
log "Base optimizer: lr=$BASE_LEARNING_RATE wd=$BASE_WEIGHT_DECAY beta2=$BASE_BETA2 grad_clip_norm=$BASE_GRAD_CLIP_NORM warmup_steps=$BASE_WARMUP_STEPS"
log "Dynamic search plan: order=$SEARCH_PARAM_ORDER screen_iterations=$SCREEN_ITERATIONS long_iterations=$LONG_ITERATIONS improvement_threshold=$IMPROVEMENT_THRESHOLD worse_guard_threshold=$WORSE_GUARD_THRESHOLD current_full_val_bpb=$CURRENT_FULL_VAL_BPB"

CONTROL_RUN_ID="$(build_run_id "dynamic_control" "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS")"
if ! run_train \
  "stage_f_p5_control" \
  "$CONTROL_RUN_ID" \
  "$CURRENT_LEARNING_RATE" \
  "$CURRENT_WEIGHT_DECAY" \
  "$CURRENT_BETA2" \
  "$CURRENT_GRAD_CLIP_NORM" \
  "$CURRENT_WARMUP_STEPS" \
  "$SCREEN_ITERATIONS"; then
  append_summary_row \
    "stage_f_p5_control" "$CONTROL_RUN_ID" "$LAST_RUN_STATUS" "0" "control" "0" "" "" "rejected" "control_failed" \
    "" "" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$SCREEN_ITERATIONS" \
    "$LAST_RUN_LAST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
  write_search_state
  log "Batch completed with failure during control run. Summary: $SUMMARY_TSV"
  exit 1
fi

CONTROL_METRIC_INFO=()
while IFS= read -r line; do
  CONTROL_METRIC_INFO+=("$line")
done < <(metric_info_for_run \
  "$CONTROL_RUN_ID" \
  "$CURRENT_LEARNING_RATE" \
  "$CURRENT_WEIGHT_DECAY" \
  "$CURRENT_BETA2" \
  "$CURRENT_GRAD_CLIP_NORM" \
  "$CURRENT_WARMUP_STEPS" \
  "$SCREEN_ITERATIONS" 2>>"$RUNNER_LOG")

if [[ "${#CONTROL_METRIC_INFO[@]}" -lt 1 || -z "${CONTROL_METRIC_INFO[0]}" ]]; then
  append_summary_row \
    "stage_f_p5_control" "$CONTROL_RUN_ID" "invalid" "0" "control" "0" "" "" "rejected" "missing_metric" \
    "" "" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$SCREEN_ITERATIONS" \
    "$LAST_RUN_LAST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
  write_search_state
  log "Batch completed with failure while reading the control metric. Summary: $SUMMARY_TSV"
  exit 1
fi

CURRENT_RUN_ID="$CONTROL_RUN_ID"
CURRENT_SCREEN_METRIC="${CONTROL_METRIC_INFO[0]}"
append_summary_row \
  "stage_f_p5_control" "$CONTROL_RUN_ID" "$LAST_RUN_STATUS" "0" "control" "0" "" "" "accepted" "control_incumbent" \
  "" "$CURRENT_SCREEN_METRIC" "$CURRENT_SCREEN_METRIC" "" "" \
  "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$SCREEN_ITERATIONS" \
  "$LAST_RUN_LAST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
write_search_state
log "Control incumbent: run_id=$CURRENT_RUN_ID metric=$CURRENT_SCREEN_METRIC"

IFS=',' read -r -a PARAM_ORDER <<< "$SEARCH_PARAM_ORDER"

PASS_NUM=1
while (( PASS_NUM <= MAX_SEARCH_PASSES )); do
  PASS_ACCEPTED_CHANGES=0
  log "Starting coordinate-search pass=$PASS_NUM incumbent_run_id=$CURRENT_RUN_ID incumbent_metric=$CURRENT_SCREEN_METRIC"

  for param in "${PARAM_ORDER[@]}"; do
    param="${param// /}"
    [[ -z "$param" ]] && continue
    if ! search_parameter "$PASS_NUM" "$param"; then
      log "Batch completed with failure during pass=$PASS_NUM parameter=$param. Summary: $SUMMARY_TSV"
      exit 1
    fi
  done

  log "Completed coordinate-search pass=$PASS_NUM accepted_changes=$PASS_ACCEPTED_CHANGES incumbent_run_id=$CURRENT_RUN_ID incumbent_metric=$CURRENT_SCREEN_METRIC"
  write_search_state

  if (( PASS_ACCEPTED_CHANGES == 0 )); then
    break
  fi

  PASS_NUM=$((PASS_NUM + 1))
done

if (( TOTAL_ACCEPTED_CHANGES == 0 )); then
  log "Dynamic search found no accepted changes; promoting the current P5 recipe directly to the longer run"
else
  log "Dynamic search converged with $TOTAL_ACCEPTED_CHANGES accepted changes; final incumbent run_id=$CURRENT_RUN_ID metric=$CURRENT_SCREEN_METRIC lr=$CURRENT_LEARNING_RATE wd=$CURRENT_WEIGHT_DECAY beta2=$CURRENT_BETA2 grad_clip_norm=$CURRENT_GRAD_CLIP_NORM warmup_steps=$CURRENT_WARMUP_STEPS"
fi

FINAL_LONG_RUN_ID="$(build_run_id "dynamic_final" "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS")_long${LONG_ITERATIONS}"
if ! run_train \
  "stage_f_p5_long" \
  "$FINAL_LONG_RUN_ID" \
  "$CURRENT_LEARNING_RATE" \
  "$CURRENT_WEIGHT_DECAY" \
  "$CURRENT_BETA2" \
  "$CURRENT_GRAD_CLIP_NORM" \
  "$CURRENT_WARMUP_STEPS" \
  "$LONG_ITERATIONS"; then
  append_summary_row \
    "stage_f_p5_long" "$FINAL_LONG_RUN_ID" "$LAST_RUN_STATUS" "$PASS_NUM" "final_recipe" "0" "" "" "rejected" "long_run_failed" \
    "$CURRENT_SCREEN_METRIC" "$CURRENT_SCREEN_METRIC" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
    "$LAST_RUN_LAST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
  write_search_state
  log "Batch completed with failure during the final long run. Summary: $SUMMARY_TSV"
  exit 1
fi

LONG_METRIC_INFO=()
while IFS= read -r line; do
  LONG_METRIC_INFO+=("$line")
done < <(metric_info_for_run \
  "$FINAL_LONG_RUN_ID" \
  "$CURRENT_LEARNING_RATE" \
  "$CURRENT_WEIGHT_DECAY" \
  "$CURRENT_BETA2" \
  "$CURRENT_GRAD_CLIP_NORM" \
  "$CURRENT_WARMUP_STEPS" \
  "$LONG_ITERATIONS" 2>>"$RUNNER_LOG")

FINAL_LONG_RUN_METRIC="${LONG_METRIC_INFO[0]:-}"
append_summary_row \
  "stage_f_p5_long" "$FINAL_LONG_RUN_ID" "$LAST_RUN_STATUS" "$PASS_NUM" "final_recipe" "0" "" "" "accepted" "long_run_launched" \
  "$CURRENT_SCREEN_METRIC" "" "$FINAL_LONG_RUN_METRIC" "" "" \
  "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
  "$LAST_RUN_LAST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
write_search_state

if [[ "$RUN_FULL_EVAL" != "1" ]]; then
  log "Skipping full eval because RUN_FULL_EVAL=$RUN_FULL_EVAL"
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  FINAL_FULL_EVAL_STATUS="dry-run"
  append_summary_row \
    "stage_f_p5_full_eval" "$FINAL_LONG_RUN_ID" "dry-run" "$PASS_NUM" "final_recipe" "0" "" "" "rejected" "full_eval_skipped_in_dry_run" \
    "$CURRENT_FULL_VAL_BPB" "" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
    "$LAST_RUN_BEST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
  write_search_state
  log "Batch completed successfully. Summary: $SUMMARY_TSV"
  exit 0
fi

BEST_LONG_CHECKPOINT="$(best_checkpoint_for_run "$FINAL_LONG_RUN_ID" 2>>"$RUNNER_LOG")" || {
  append_summary_row \
    "stage_f_p5_full_eval" "$FINAL_LONG_RUN_ID" "failed" "$PASS_NUM" "final_recipe" "0" "" "" "rejected" "missing_best_checkpoint" \
    "$CURRENT_FULL_VAL_BPB" "" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
    "$LAST_RUN_BEST_CKPT" "$LAST_RUN_BEST_CKPT" "$LAST_RUN_LOG_PATH"
  write_search_state
  log "Could not determine the best checkpoint for $FINAL_LONG_RUN_ID"
  exit 1
}

if ! run_full_eval \
  "stage_f_p5_full_eval" \
  "$FINAL_LONG_RUN_ID" \
  "$BEST_LONG_CHECKPOINT" \
  "$CURRENT_LEARNING_RATE" \
  "$CURRENT_WEIGHT_DECAY" \
  "$CURRENT_BETA2" \
  "$CURRENT_GRAD_CLIP_NORM" \
  "$CURRENT_WARMUP_STEPS"; then
  FINAL_FULL_EVAL_STATUS="$LAST_FULL_EVAL_STATUS"
  append_summary_row \
    "stage_f_p5_full_eval" "$FINAL_LONG_RUN_ID" "$LAST_FULL_EVAL_STATUS" "$PASS_NUM" "final_recipe" "0" "" "" "rejected" "full_eval_failed" \
    "$CURRENT_FULL_VAL_BPB" "" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
    "$BEST_LONG_CHECKPOINT" "$BEST_LONG_CHECKPOINT" "$LAST_FULL_EVAL_LOG_PATH"
  write_search_state
  exit 1
fi

FINAL_FULL_EVAL_STATUS="$LAST_FULL_EVAL_STATUS"
FINAL_FULL_EVAL_BPB="$(full_eval_bpb_from_log "$LAST_FULL_EVAL_LOG_PATH" 2>>"$RUNNER_LOG")" || {
  append_summary_row \
    "stage_f_p5_full_eval" "$FINAL_LONG_RUN_ID" "invalid" "$PASS_NUM" "final_recipe" "0" "" "" "rejected" "missing_full_eval_metric" \
    "$CURRENT_FULL_VAL_BPB" "" "" "" "" \
    "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
    "$BEST_LONG_CHECKPOINT" "$BEST_LONG_CHECKPOINT" "$LAST_FULL_EVAL_LOG_PATH"
  write_search_state
  log "Batch completed with failure while reading the full-eval metric. Summary: $SUMMARY_TSV"
  exit 1
}

FULL_EVAL_DECISION="rejected"
FULL_EVAL_REASON="full_val_not_better"
if float_le "$FINAL_FULL_EVAL_BPB" "$(float_sub "$CURRENT_FULL_VAL_BPB" "0.0000000001")"; then
  FULL_EVAL_DECISION="accepted"
  FULL_EVAL_REASON="full_val_better"
fi

append_summary_row \
  "stage_f_p5_full_eval" "$FINAL_LONG_RUN_ID" "$LAST_FULL_EVAL_STATUS" "$PASS_NUM" "final_recipe" "0" "" "" "$FULL_EVAL_DECISION" "$FULL_EVAL_REASON" \
  "$CURRENT_FULL_VAL_BPB" "$FINAL_FULL_EVAL_BPB" "" "" "$FINAL_FULL_EVAL_BPB" \
  "$CURRENT_LEARNING_RATE" "$CURRENT_WEIGHT_DECAY" "$CURRENT_BETA2" "$CURRENT_GRAD_CLIP_NORM" "$CURRENT_WARMUP_STEPS" "$LONG_ITERATIONS" \
  "$BEST_LONG_CHECKPOINT" "$BEST_LONG_CHECKPOINT" "$LAST_FULL_EVAL_LOG_PATH"
write_search_state

log "Full eval result for run_id=$FINAL_LONG_RUN_ID val_bpb=$FINAL_FULL_EVAL_BPB benchmark=$CURRENT_FULL_VAL_BPB decision=$FULL_EVAL_REASON"
log "Batch completed successfully. Summary: $SUMMARY_TSV"
exit 0
