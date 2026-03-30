#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

MODE="search"
CONFIG="search_configs/metastack_v2_wd_sliding_remote.yaml"
WORKDIR="/workspace/parameter-golf"
VARIANT="sp1024"
TRAIN_SHARDS="80"
MAX_RUNS="2"
WORLD_SIZE="8"
LAUNCH_ID="$(date -u +"%Y%m%dT%H%M%SZ")"
REMOTE_PYTHON_BIN=""
MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-willdepueoai/parameter-golf}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --train-shards) TRAIN_SHARDS="$2"; shift 2 ;;
        --max-runs) MAX_RUNS="$2"; shift 2 ;;
        --world-size) WORLD_SIZE="$2"; shift 2 ;;
        --launch-id) LAUNCH_ID="$2"; shift 2 ;;
        --prebuilt-python-bin) REMOTE_PYTHON_BIN="$2"; shift 2 ;;
        --matched-fineweb-repo-id) MATCHED_FINEWEB_REPO_ID="$2"; shift 2 ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

RUN_ROOT="$WORKDIR/deploy_runs/$LAUNCH_ID"
LOG_DIR="$RUN_ROOT/logs"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
STATE_DIR="$RUN_ROOT/state"
WORKLOAD_LOG_DIR="$ARTIFACT_DIR/workload_logs"
SEARCH_OUTPUT_ROOT="$ARTIFACT_DIR/search_output"
RENDERED_CONFIG="$ARTIFACT_DIR/rendered_config.yaml"
BOOTSTRAP_LOG="$LOG_DIR/bootstrap.log"
COMMAND_LOG="$LOG_DIR/commands.tsv"
GPU_MONITOR_LOG="$LOG_DIR/gpu_metrics.csv"
SYSTEM_MONITOR_LOG="$LOG_DIR/system_metrics.csv"
STATUS_JSON="$STATE_DIR/status.json"
SUMMARY_JSON="$STATE_DIR/summary.json"
PYTHON_BIN=""
PIP_BIN=""
PTXAS_PATH=""
STARTED_AT="$(timestamp)"
FINAL_STATUS="running"
FAILURE_MESSAGE=""
CURRENT_PHASE="bootstrap"
GPU_MONITOR_PID=""
SYSTEM_MONITOR_PID=""

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$STATE_DIR" "$WORKLOAD_LOG_DIR" "$SEARCH_OUTPUT_ROOT"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

log() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

write_status() {
    local phase="$1"
    local status="$2"
    local message="${3:-}"
    python3 - "$STATUS_JSON" "$phase" "$status" "$message" "$LAUNCH_ID" "$MODE" "$WORKDIR" "$CONFIG" "$RENDERED_CONFIG" "$RUN_ROOT" "$PYTHON_BIN" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
phase, status, message, launch_id, mode, workdir, config, rendered_config, run_root, python_bin = sys.argv[2:]
payload = {
    "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "phase": phase,
    "status": status,
    "message": message,
    "launch_id": launch_id,
    "mode": mode,
    "workdir": workdir,
    "config": config,
    "rendered_config": rendered_config,
    "run_root": run_root,
    "python_bin": python_bin,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_logged() {
    local label="$1"
    shift
    local started_iso started_epoch finished_epoch finished_iso duration rc
    started_iso="$(timestamp)"
    started_epoch="$(date +%s)"
    log "BEGIN: $label"
    set +e
    "$@"
    rc=$?
    set -e
    finished_epoch="$(date +%s)"
    finished_iso="$(timestamp)"
    duration=$((finished_epoch - started_epoch))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$started_iso" "$finished_iso" "$rc" "$duration" "$CURRENT_PHASE" "$label" >> "$COMMAND_LOG"
    log "END rc=$rc duration=${duration}s: $label"
    return "$rc"
}

stop_monitors() {
    if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
    if [[ -n "$SYSTEM_MONITOR_PID" ]] && kill -0 "$SYSTEM_MONITOR_PID" 2>/dev/null; then
        kill "$SYSTEM_MONITOR_PID" 2>/dev/null || true
        wait "$SYSTEM_MONITOR_PID" 2>/dev/null || true
    fi
}

start_monitors() {
    (
        printf 'timestamp,index,name,util_gpu,util_mem,mem_used_mb,mem_total_mb,power_w,temp_c\n'
        while true; do
            local now
            now="$(timestamp)"
            while IFS=, read -r idx name util_gpu util_mem mem_used mem_total power temp; do
                [[ -z "${idx:-}" ]] && continue
                printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$now" "$idx" "$name" "$util_gpu" "$util_mem" "$mem_used" "$mem_total" "$power" "$temp"
            done < <(nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)
            sleep 5
        done
    ) >"$GPU_MONITOR_LOG" &
    GPU_MONITOR_PID="$!"

    (
        printf 'timestamp,fs_size_bytes,fs_used_bytes,fs_avail_bytes,mem_total_bytes,mem_available_bytes,load1\n'
        while true; do
            local now fs size used avail mem_total mem_avail load1
            now="$(timestamp)"
            read -r size used avail < <(df -B1 --output=size,used,avail "$WORKDIR" | awk 'NR==2 {print $1, $2, $3}')
            read -r mem_total mem_avail < <(free -b | awk '/^Mem:/ {print $2, $7}')
            read -r load1 _ < /proc/loadavg
            printf '%s,%s,%s,%s,%s,%s,%s\n' "$now" "$size" "$used" "$avail" "$mem_total" "$mem_avail" "$load1"
            sleep 15
        done
    ) >"$SYSTEM_MONITOR_LOG" &
    SYSTEM_MONITOR_PID="$!"
}

write_summary() {
    local exit_code="$1"
    local finished_at="$2"
    python3 - "$SUMMARY_JSON" "$LAUNCH_ID" "$MODE" "$CONFIG" "$RENDERED_CONFIG" "$RUN_ROOT" "$WORKDIR" "$PYTHON_BIN" "$PTXAS_PATH" "$MATCHED_FINEWEB_REPO_ID" "$VARIANT" "$TRAIN_SHARDS" "$MAX_RUNS" "$WORLD_SIZE" "$FINAL_STATUS" "$FAILURE_MESSAGE" "$STARTED_AT" "$finished_at" "$exit_code" "$SEARCH_OUTPUT_ROOT" "$WORKLOAD_LOG_DIR" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    launch_id,
    mode,
    config,
    rendered_config,
    run_root,
    workdir,
    python_bin,
    ptxas_path,
    repo_id,
    variant,
    train_shards,
    max_runs,
    world_size,
    status,
    failure_message,
    started_at,
    finished_at,
    exit_code,
    search_output_root,
    workload_log_dir,
) = sys.argv[1:]

payload = {
    "launch_id": launch_id,
    "mode": mode,
    "config": config,
    "rendered_config": rendered_config,
    "run_root": run_root,
    "workdir": workdir,
    "python_bin": python_bin,
    "ptxas_path": ptxas_path,
    "matched_fineweb_repo_id": repo_id,
    "variant": variant,
    "train_shards": int(train_shards),
    "max_runs": None if max_runs == "" else int(max_runs),
    "world_size": int(world_size),
    "status": status,
    "failure_message": failure_message,
    "started_at": started_at,
    "finished_at": finished_at,
    "exit_code": int(exit_code),
    "search_output_root": search_output_root,
    "workload_log_dir": workload_log_dir,
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

on_error() {
    local line="$1"
    local cmd="$2"
    local rc="$3"
    FAILURE_MESSAGE="line=${line} cmd=${cmd} rc=${rc}"
    FINAL_STATUS="failed"
    write_status "$CURRENT_PHASE" "$FINAL_STATUS" "$FAILURE_MESSAGE"
}

on_exit() {
    local rc="$?"
    local finished_at
    finished_at="$(timestamp)"
    stop_monitors
    if [[ "$FINAL_STATUS" != "failed" ]]; then
        if [[ "$rc" -eq 0 ]]; then
            FINAL_STATUS="completed"
        else
            FINAL_STATUS="failed"
            FAILURE_MESSAGE="script exited with rc=$rc"
        fi
    fi
    write_status "$CURRENT_PHASE" "$FINAL_STATUS" "$FAILURE_MESSAGE"
    write_summary "$rc" "$finished_at"
    if [[ "$FINAL_STATUS" == "completed" ]]; then
        log "COMPLETED: launch_id=$LAUNCH_ID"
    else
        log "FAILED: $FAILURE_MESSAGE"
    fi
}

trap 'on_error "$LINENO" "$BASH_COMMAND" "$?"' ERR
trap 'on_exit' EXIT

resolve_ptxas() {
    if [[ -x /usr/local/cuda/bin/ptxas ]]; then
        printf '%s\n' /usr/local/cuda/bin/ptxas
        return
    fi
    if command -v ptxas >/dev/null 2>&1; then
        command -v ptxas
        return
    fi
    return 1
}

preflight() {
    CURRENT_PHASE="preflight"
    write_status "$CURRENT_PHASE" "running" "verifying remote host"
    run_logged "uname -a" uname -a
    run_logged "python3 --version" python3 --version
    run_logged "nvidia-smi" nvidia-smi
    run_logged "nvidia-smi -L" bash -lc 'nvidia-smi -L | tee "$0"' "$ARTIFACT_DIR/nvidia-smi-L.txt"
    run_logged "nvidia-smi topo -m" bash -lc 'nvidia-smi topo -m | tee "$0"' "$ARTIFACT_DIR/nvidia-smi-topo.txt"
    run_logged "lscpu" bash -lc 'lscpu | tee "$0"' "$ARTIFACT_DIR/lscpu.txt"
    run_logged "df -h" bash -lc 'df -h | tee "$0"' "$ARTIFACT_DIR/df-h.txt"
    run_logged "ulimit -a" bash -lc 'ulimit -a | tee "$0"' "$ARTIFACT_DIR/ulimit.txt"

    local arch gpu_count
    arch="$(uname -m)"
    [[ "$arch" == "x86_64" || "$arch" == "amd64" ]] || { echo "Expected amd64/x86_64, got $arch"; return 1; }
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    [[ "$gpu_count" == "$WORLD_SIZE" ]] || { echo "Expected $WORLD_SIZE GPUs, found $gpu_count"; return 1; }
    nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'H100' || {
        echo "Expected H100 GPUs"
        return 1
    }
    PTXAS_PATH="$(resolve_ptxas)"
    [[ -n "$PTXAS_PATH" ]] || { echo "Could not resolve ptxas"; return 1; }
    log "Resolved ptxas: $PTXAS_PATH"
}

setup_python() {
    CURRENT_PHASE="python_bootstrap"
    write_status "$CURRENT_PHASE" "running" "bootstrapping remote python env"

    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_INPUT=1
    export PIP_CACHE_DIR="$WORKDIR/.cache/pip"
    export HF_HOME="$WORKDIR/.cache/huggingface"
    mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

    if [[ -n "$REMOTE_PYTHON_BIN" ]]; then
        [[ -x "$REMOTE_PYTHON_BIN" ]] || { echo "prebuilt python not executable: $REMOTE_PYTHON_BIN"; return 1; }
        PYTHON_BIN="$REMOTE_PYTHON_BIN"
    else
        PYTHON_BIN="$WORKDIR/.venv-remote/bin/python"
        if [[ ! -x "$PYTHON_BIN" ]]; then
            run_logged "create venv" python3 -m venv --system-site-packages "$WORKDIR/.venv-remote"
        else
            log "Reusing existing venv at $WORKDIR/.venv-remote"
        fi
    fi
    PIP_BIN="$(dirname "$PYTHON_BIN")/pip"

    local pip_retry=0
    until run_logged "pip install runtime deps" "$PIP_BIN" install --upgrade pip wheel setuptools -r "$WORKDIR/deploy/vast/requirements.remote.txt"; do
        pip_retry=$((pip_retry + 1))
        [[ "$pip_retry" -lt 3 ]] || return 1
        sleep $((pip_retry * 5))
    done

    if "$PYTHON_BIN" -c 'import hf_transfer' >/dev/null 2>&1; then
        export HF_HUB_ENABLE_HF_TRANSFER=1
        log "Enabled HF_HUB_ENABLE_HF_TRANSFER=1"
    fi

    run_logged "python import smoke" "$PYTHON_BIN" - <<'PY'
import huggingface_hub
import numpy
import scipy
import sentencepiece
import sklearn
import torch
import yaml
import zstandard

print("imports_ok")
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
PY

    run_logged "pip freeze" bash -lc '"$0" -m pip freeze | tee "$1"' "$PYTHON_BIN" "$ARTIFACT_DIR/pip-freeze.txt"
}

run_ddp_smoke() {
    CURRENT_PHASE="ddp_smoke"
    write_status "$CURRENT_PHASE" "running" "running 8-GPU DDP smoke"
    run_logged "torch distributed smoke" "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" "$WORKDIR/deploy/vast/ddp_smoke.py"
}

run_unit_tests() {
    CURRENT_PHASE="unit_tests"
    write_status "$CURRENT_PHASE" "running" "running deployment unit tests"
    run_logged "unit tests" "$PYTHON_BIN" -m unittest tests.test_search_runner tests.test_vast_render_remote_config
}

fetch_data() {
    CURRENT_PHASE="data_fetch"
    write_status "$CURRENT_PHASE" "running" "downloading challenge data"
    run_logged "cached challenge fineweb" env MATCHED_FINEWEB_REPO_ID="$MATCHED_FINEWEB_REPO_ID" "$PYTHON_BIN" "$WORKDIR/data/cached_challenge_fineweb.py" --variant "$VARIANT" --train-shards "$TRAIN_SHARDS"
    run_logged "dataset inventory" bash -lc 'find "$0/data" -maxdepth 3 -type f | sort | tee "$1"' "$WORKDIR" "$ARTIFACT_DIR/data-files.txt"
}

render_config() {
    CURRENT_PHASE="render_config"
    write_status "$CURRENT_PHASE" "running" "rendering remote search config"
    local config_stem output_root_rel
    config_stem="$(basename "$CONFIG")"
    config_stem="${config_stem%.yaml}"
    output_root_rel="deploy_runs/${LAUNCH_ID}/artifacts/search_output"
    run_logged "render remote config" "$PYTHON_BIN" "$WORKDIR/deploy/vast/render_remote_config.py" \
        --input "$WORKDIR/$CONFIG" \
        --output "$RENDERED_CONFIG" \
        --workdir "$WORKDIR" \
        --python-bin "$PYTHON_BIN" \
        --gpus "$WORLD_SIZE" \
        --logs-dir "$WORKLOAD_LOG_DIR" \
        --output-root "$output_root_rel" \
        --set-fixed-env "TRITON_PTXAS_PATH=$PTXAS_PATH"
}

run_workload() {
    case "$MODE" in
        bootstrap-only)
            CURRENT_PHASE="bootstrap_only"
            write_status "$CURRENT_PHASE" "running" "bootstrap-only mode complete"
            ;;
        search)
            CURRENT_PHASE="search"
            write_status "$CURRENT_PHASE" "running" "launching search workload"
            if [[ -n "$MAX_RUNS" ]]; then
                run_logged "search workload" "$PYTHON_BIN" "$WORKDIR/search/run_search.py" --config "$RENDERED_CONFIG" --max-runs "$MAX_RUNS"
            else
                run_logged "search workload" "$PYTHON_BIN" "$WORKDIR/search/run_search.py" --config "$RENDERED_CONFIG"
            fi
            ;;
        *)
            echo "unsupported mode: $MODE" >&2
            return 1
            ;;
    esac
}

main() {
    printf 'started_at\tfinished_at\trc\tduration_seconds\tphase\tlabel\n' > "$COMMAND_LOG"
    write_status "$CURRENT_PHASE" "running" "bootstrap starting"
    start_monitors
    preflight
    setup_python
    run_ddp_smoke
    run_unit_tests
    fetch_data
    render_config
    run_workload
}

main
