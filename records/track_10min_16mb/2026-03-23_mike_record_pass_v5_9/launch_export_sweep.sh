#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/launch_config.json"
CHECKPOINT=""
OUTPUT_DIR=""
POLICY_ID="v5_9_t0_export"
BUDGETS=""
TRAINING_INSTABILITY=0
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --policy-id)
      POLICY_ID="$2"
      shift 2
      ;;
    --budgets)
      BUDGETS="$2"
      shift 2
      ;;
    --training-instability)
      TRAINING_INSTABILITY=1
      shift
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cfg_get() {
  local key="$1"
  python3 - "$CONFIG" "$key" <<'PY'
import json
import os
import pathlib
import subprocess
import sys

config_path = pathlib.Path(sys.argv[1]).resolve()
key = sys.argv[2]
cfg = json.loads(config_path.read_text(encoding="utf-8"))

def detect_repo_root(path):
    env_repo_root = os.environ.get("REPO_ROOT")
    if env_repo_root:
        return pathlib.Path(env_repo_root).expanduser().resolve()
    try:
        proc = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = pathlib.Path(proc.stdout.strip()).resolve()
        if candidate.exists():
            return candidate
    except Exception:
        pass
    for candidate in [path.parent, *path.parents]:
        if (candidate / "analysis").exists() and (candidate / "records").exists():
            return candidate
    return path.parents[3] if len(path.parents) > 3 else path.parent

repo_root = detect_repo_root(config_path)
record_dir = pathlib.Path(os.environ.get("RECORD_DIR", str(config_path.parent))).expanduser().resolve()

def expand(v):
    if isinstance(v, str):
        return v.replace("${REPO_ROOT}", str(repo_root)).replace("${RECORD_DIR}", str(record_dir))
    if isinstance(v, list):
        return [expand(x) for x in v]
    if isinstance(v, dict):
        return {k: expand(x) for k, x in v.items()}
    return v

cur = expand(cfg)
for part in key.split('.'):
    cur = cur[part]
if isinstance(cur, list):
    print(",".join(str(x) for x in cur))
else:
    print(cur)
PY
}

RECORD_DIR="$(cfg_get paths.record_dir)"
TRAIN_SCRIPT="$(cfg_get paths.train_script)"
WRAPPER_SCRIPT="$(cfg_get paths.export_wrapper_script)"
BASE_POLICY="$(cfg_get paths.base_policy_json)"
RANKING="$(cfg_get paths.promotion_ranking_csv)"
LOCALBENCH_DATA="$(cfg_get paths.localbench_data)"
TOKENIZER_PATH="$(cfg_get paths.tokenizer_path)"
OUTPUT_ROOT="$(cfg_get paths.output_root)"
ALLOW_GROUPS="$(cfg_get export.allow_promotion_groups)"
TRAIN_SEQ_LEN="$(cfg_get export.train_seq_len)"
EVAL_STRIDE="$(cfg_get export.eval_stride)"
EVAL_BATCH_SEQS="$(cfg_get export.eval_batch_seqs)"
EVAL_MAX_WINDOWS="$(cfg_get export.eval_max_windows)"
CAP_MIN="$(cfg_get export.target_cap_total_bytes_min)"
CAP_MAX="$(cfg_get export.target_cap_total_bytes_max)"
NOISE_FLOOR="$(cfg_get decision_thresholds.noise_floor_bpb)"
REF_BEST="$(cfg_get decision_thresholds.ref_best_bpb)"
SUCCESS_CUTOFF="$(cfg_get decision_thresholds.success_cutoff_bpb)"
T1_TRIGGER="$(cfg_get decision_thresholds.t1_trigger_cutoff_bpb)"
CORE_BUDGETS="$(cfg_get export.core_budgets)"
LARGER_BUDGETS="$(cfg_get export.larger_budgets)"

if [[ -z "$BUDGETS" ]]; then
  BUDGETS="$(cfg_get export.budgets)"
fi
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$OUTPUT_ROOT/T0/final_model.pt"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$OUTPUT_ROOT/T0"
fi

mkdir -p "$OUTPUT_DIR"

EXPORT_SUMMARY_CSV="$OUTPUT_DIR/export_summary.csv"
SERIALIZER_CSV="$OUTPUT_DIR/serializer_trials.csv"
DECISION_JSON="$OUTPUT_DIR/export_decision.json"
MANIFEST_JSON="$OUTPUT_DIR/export_manifest.json"
ARTIFACT_PATH="$OUTPUT_DIR/final_model.recommended.int8.ptz"
LOG_FILE="$OUTPUT_DIR/export_sweep_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  python3 "$WRAPPER_SCRIPT"
  --checkpoint "$CHECKPOINT"
  --train-script "$TRAIN_SCRIPT"
  --base-policy-json "$BASE_POLICY"
  --promotion-ranking-csv "$RANKING"
  --promotion-budget-bytes-list "$BUDGETS"
  --allow-promotion-groups "$ALLOW_GROUPS"
  --policy-id "$POLICY_ID"
  --eval-roundtrip
  --data-path "$LOCALBENCH_DATA"
  --tokenizer-path "$TOKENIZER_PATH"
  --train-seq-len "$TRAIN_SEQ_LEN"
  --eval-stride "$EVAL_STRIDE"
  --eval-batch-seqs "$EVAL_BATCH_SEQS"
  --eval-max-windows "$EVAL_MAX_WINDOWS"
  --target-cap-total-bytes-min "$CAP_MIN"
  --target-cap-total-bytes-max "$CAP_MAX"
  --noise-floor-bpb "$NOISE_FLOOR"
  --ref-best-bpb "$REF_BEST"
  --success-cutoff-bpb "$SUCCESS_CUTOFF"
  --t1-trigger-cutoff-bpb "$T1_TRIGGER"
  --core-budgets "$CORE_BUDGETS"
  --larger-budgets "$LARGER_BUDGETS"
  --write-artifact "$ARTIFACT_PATH"
  --output-launchcheck-csv "$EXPORT_SUMMARY_CSV"
  --output-decision-json "$DECISION_JSON"
  --output-manifest-json "$MANIFEST_JSON"
  --output-serializer-csv "$SERIALIZER_CSV"
)

if [[ "$TRAINING_INSTABILITY" -eq 1 ]]; then
  CMD+=(--training-instability)
fi

cat <<CONF
[launch_export_sweep] canonical configuration
config=$CONFIG
record_dir=$RECORD_DIR
checkpoint=$CHECKPOINT
output_dir=$OUTPUT_DIR
policy_id=$POLICY_ID
budgets=$BUDGETS
summary_csv=$EXPORT_SUMMARY_CSV
serializer_csv=$SERIALIZER_CSV
decision_json=$DECISION_JSON
artifact_path=$ARTIFACT_PATH
log_file=$LOG_FILE
command=${CMD[*]}
CONF

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  exit 0
fi

(
  cd "$RECORD_DIR"
  "${CMD[@]}"
) 2>&1 | tee "$LOG_FILE"
export_exit=${PIPESTATUS[0]}
if [[ "$export_exit" -ne 0 ]]; then
  echo "[launch_export_sweep] export sweep failed with exit code $export_exit" >&2
  exit "$export_exit"
fi

if [[ ! -f "$DECISION_JSON" ]]; then
  echo "[launch_export_sweep] missing decision JSON: $DECISION_JSON" >&2
  exit 4
fi

python3 - "$DECISION_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
payload = json.loads(path.read_text(encoding="utf-8"))
rec = payload.get("recommended_option", {})
decision = payload.get("decision_block", {}).get("stop_rule_outcome", "UNKNOWN")
print(
    "[launch_export_sweep] recommended "
    f"budget={rec.get('budget_bytes', 'NA')} "
    f"serializer={rec.get('serializer_variant', 'NA')} "
    f"scale={rec.get('scale_mode', 'NA')} "
    f"compressor={rec.get('compressor', 'NA')} "
    f"decision={decision}"
)
PY
