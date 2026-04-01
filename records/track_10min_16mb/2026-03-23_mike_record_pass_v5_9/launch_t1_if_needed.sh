#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/launch_config.json"
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
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

export_common_env() {
  python3 - "$CONFIG" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1]).resolve()
cfg = json.loads(config_path.read_text(encoding="utf-8"))
env_map = cfg["training"]["common_env"]
for key in sorted(env_map):
    print(f"{key}={env_map[key]}")
PY
}

RECORD_DIR="$(cfg_get paths.record_dir)"
TRAIN_SCRIPT="$(cfg_get paths.train_script)"
DATA_PATH="$(cfg_get paths.data_path)"
TOKENIZER_PATH="$(cfg_get paths.tokenizer_path)"
BASE_POLICY="$(cfg_get paths.base_policy_json)"
OUTPUT_ROOT="$(cfg_get paths.output_root)"
RUN_ID="$(cfg_get training.run_ids.t1)"
SEED="$(cfg_get training.seeds.t1)"
NPROC_PER_NODE="$(cfg_get hardware_expectations.expected_cuda_devices)"

OUT_DIR="$OUTPUT_ROOT/T1"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"

while IFS='=' read -r k v; do
  export "$k=$v"
done < <(export_common_env)

export RUN_ID
export SEED
export DATA_PATH
export TOKENIZER_PATH
export PRECISION_OVERRIDE_PATH="$BASE_POLICY"

TRAIN_CMD=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$TRAIN_SCRIPT")
EXPORT_CMD=("$SCRIPT_DIR/launch_export_sweep.sh" --config "$CONFIG" --checkpoint "$OUT_DIR/final_model.pt" --output-dir "$OUT_DIR" --policy-id "v5_9_t1_export")

cat <<CONF
[launch_t1_if_needed] canonical configuration
config=$CONFIG
record_dir=$RECORD_DIR
output_dir=$OUT_DIR
log_file=$LOG_FILE
run_id=$RUN_ID
seed=$SEED
data_path=$DATA_PATH
tokenizer_path=$TOKENIZER_PATH
precision_override_path=$PRECISION_OVERRIDE_PATH
train_command=${TRAIN_CMD[*]}
export_command=${EXPORT_CMD[*]}
CONF

if [[ "$PRINT_ONLY" -eq 1 ]]; then
  "$SCRIPT_DIR/launch_export_sweep.sh" --config "$CONFIG" --checkpoint "$OUT_DIR/final_model.pt" --output-dir "$OUT_DIR" --policy-id "v5_9_t1_export" --print-only
  exit 0
fi

(
  cd "$RECORD_DIR"
  "${TRAIN_CMD[@]}"
) 2>&1 | tee "$LOG_FILE"
train_exit=${PIPESTATUS[0]}
if [[ "$train_exit" -ne 0 ]]; then
  echo "[launch_t1_if_needed] training failed with exit code $train_exit" >&2
  exit "$train_exit"
fi

if [[ ! -f "$RECORD_DIR/final_model.pt" ]]; then
  echo "[launch_t1_if_needed] missing checkpoint: $RECORD_DIR/final_model.pt" >&2
  exit 3
fi

cp "$RECORD_DIR/final_model.pt" "$OUT_DIR/final_model.pt"
echo "[launch_t1_if_needed] checkpoint copied to $OUT_DIR/final_model.pt"

"${EXPORT_CMD[@]}"
