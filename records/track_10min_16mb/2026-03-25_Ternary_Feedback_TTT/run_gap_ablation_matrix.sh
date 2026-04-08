#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"
BALANCE_FLOOR="${BALANCE_FLOOR:-50}"
GPU_COUNT="${GPU_COUNT:-1}"
DATA_SHARDS="${DATA_SHARDS:-12}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-12}"
VOLUME_GB="${VOLUME_GB:-20}"
DISK_GB="${DISK_GB:-20}"
MIN_VCPU_COUNT="${MIN_VCPU_COUNT:-1}"
MIN_SYSTEM_MEMORY_GB="${MIN_SYSTEM_MEMORY_GB:-8}"
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-12}"

OUT_ROOT="${DIR}/gap_ablation_matrix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
SUMMARY_TSV="${OUT_ROOT}/summary.tsv"

printf "label\tartifact_dir\traw_step\traw_bpb\troundtrip_bpb\tsliding_bpb\tngram_bpb\tartifact_mb\n" > "$SUMMARY_TSV"

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }

get_balance() {
  python3 - <<'PY'
import json, os, subprocess
api = os.environ["RUNPOD_API_KEY"]
query = 'query { myself { clientBalance } }'
out = subprocess.check_output([
    'curl', '-s', '--request', 'POST',
    '--header', 'content-type: application/json',
    '--url', f'https://api.runpod.io/graphql?api_key={api}',
    '--data', json.dumps({'query': query}),
], text=True)
print(json.loads(out)['data']['myself']['clientBalance'])
PY
}

balance_ok() {
  local balance="$1"
  awk -v b="$balance" -v floor="$BALANCE_FLOOR" 'BEGIN { exit !(b > floor) }'
}

append_summary() {
  local label="$1"
  local artifact_dir="$2"
  local log_path="${artifact_dir}/full_run.log"
  python3 - "$label" "$artifact_dir" "$log_path" "$SUMMARY_TSV" <<'PY'
import pathlib, re, sys
label, artifact_dir, log_path, summary_path = sys.argv[1:]
text = pathlib.Path(log_path).read_text()
def grab(pattern):
    m = re.search(pattern, text)
    return m.groups() if m else ("",)
raws = re.findall(r"step:(\d+)/500000 val_loss:([0-9.]+) val_bpb:([0-9.]+) train_time:(\d+)ms zero_frac:([0-9.]+)", text)
raw_step = raws[-1][0] if raws else ""
raw_bpb = raws[-1][2] if raws else ""
roundtrip_bpb = grab(r"final_ternary_roundtrip val_loss:[0-9.]+ val_bpb:([0-9.]+)")[0]
sliding_bpb = grab(r"final_sliding val_loss:[0-9.]+ val_bpb:([0-9.]+)")[0]
ngram_bpb = grab(r"ngram_cache val_loss:[0-9.]+ val_bpb:([0-9.]+)")[0]
artifact_mb = grab(r"budget:\d+/16000000 \(([0-9.]+)/16.00MB\) FITS")[0]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\t".join([label, artifact_dir, raw_step, raw_bpb, roundtrip_bpb, sliding_bpb, ngram_bpb, artifact_mb]) + "\n")
PY
}

run_config() {
  local label="$1"
  shift
  local run_dir="${OUT_ROOT}/${label}"
  mkdir -p "$run_dir"
  local attempt=1
  while true; do
    local balance
    balance="$(get_balance)"
    log "balance=${balance}"
    if ! balance_ok "$balance"; then
      log "Stopping matrix: balance ${balance} <= floor ${BALANCE_FLOOR}"
      return 1
    fi
    log "Starting ${label} attempt ${attempt}"
    if env \
      RUNPOD_API_KEY="$RUNPOD_API_KEY" \
      GPU_COUNT="$GPU_COUNT" \
      BALANCE_FLOOR="$BALANCE_FLOOR" \
      KEEP_POD_ON_EXIT="$KEEP_POD_ON_EXIT" \
      DATA_SHARDS="$DATA_SHARDS" \
      MIN_GPU_MEMORY_GB="$MIN_GPU_MEMORY_GB" \
      VOLUME_GB="$VOLUME_GB" \
      DISK_GB="$DISK_GB" \
      MIN_VCPU_COUNT="$MIN_VCPU_COUNT" \
      MIN_SYSTEM_MEMORY_GB="$MIN_SYSTEM_MEMORY_GB" \
      LOCAL_ARTIFACTS_DIR="$run_dir" \
      "$@" \
      bash "${DIR}/orchestrate_small_skc_multigpu_runpod.sh"; then
      append_summary "$label" "$run_dir"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP_SECONDS"
  done
}

run_config baseline \
  EXPORT_ALIGNED_TRAIN=0 \
  TERNARY_THRESHOLD_SEARCH=0 \
  TERNARY_SCALE_SEARCH=0 \
  HESSIAN_TERNARY_GPTQ=0 \
  GPTQ_LITE_ENABLED=0 \
  SELECTIVE_PRUNING=0 \
  NGRAM_CACHE_ENABLED=0 \
  TURBO_QUANT_EXPORT=1

run_config search_only \
  EXPORT_ALIGNED_TRAIN=0 \
  TERNARY_THRESHOLD_SEARCH=1 \
  TERNARY_SCALE_SEARCH=1 \
  HESSIAN_TERNARY_GPTQ=0 \
  GPTQ_LITE_ENABLED=0 \
  SELECTIVE_PRUNING=0 \
  NGRAM_CACHE_ENABLED=0 \
  TURBO_QUANT_EXPORT=1

run_config search_hessian \
  EXPORT_ALIGNED_TRAIN=0 \
  TERNARY_THRESHOLD_SEARCH=1 \
  TERNARY_SCALE_SEARCH=1 \
  HESSIAN_TERNARY_GPTQ=1 \
  GPTQ_LITE_ENABLED=0 \
  SELECTIVE_PRUNING=0 \
  NGRAM_CACHE_ENABLED=0 \
  TURBO_QUANT_EXPORT=1

run_config search_plain \
  EXPORT_ALIGNED_TRAIN=0 \
  TERNARY_THRESHOLD_SEARCH=1 \
  TERNARY_SCALE_SEARCH=1 \
  HESSIAN_TERNARY_GPTQ=0 \
  GPTQ_LITE_ENABLED=0 \
  SELECTIVE_PRUNING=0 \
  NGRAM_CACHE_ENABLED=0 \
  TURBO_QUANT_EXPORT=0

log "Matrix complete: ${SUMMARY_TSV}"
