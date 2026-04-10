#!/usr/bin/env bash
# ============================================================================
# Parameter Golf — H100 Experiment Orchestrator
# Stack : hardware-aware SKC competition line
#           starts from the simple small-SKC recipe
#           auto-scales width/batch on large multi-GPU hardware
#           no feedback / no capsule-bank / no XSA
#           LAWA+SWA(selective)  adaptive ternary export
#
# Budget  : 5400 seconds from first billable pod second
# Seeds   : default 42
# Data    : configurable train shard count downloaded from HF to pod
#
# Pipeline:
#   1. RunPod GraphQL → find + create 8×H100 SXM pod
#   2. Wait for SSH readiness
#   3. Install deps (FA3 wheel optional, sentencepiece, zstandard)
#   4. Download sp1024 data (train shards + 1 val shard) from HuggingFace
#   5. Upload code (train_gpt.py + run_h100_skc_competition.sh)
#   6. Smoke test  (FAST_SMOKE=1, MAX_WALLCLOCK_SECONDS=45)
#   7. Seed 42     (full run; default single-seed)
#   8. Download ALL artifacts + logs → local machine
#   9. Terminate pod
#
# Usage:
#   export RUNPOD_API_KEY=<your key>
#   bash orchestrate_h100_experiment.sh
# ============================================================================
set -uo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"

# ── Configurable ─────────────────────────────────────────────────────────────
LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
GPU_TYPE_ID="NVIDIA H100 80GB HBM3"
GPU_COUNT=8
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="paramgolf-$(date +%Y%m%d-%H%M%S)"
VOLUME_GB=50
DISK_GB=50
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
DATA_SHARDS="${DATA_SHARDS:-20}"   # small SKC line does not need the old 60-shard setup
HF_REPO="willdepueoai/parameter-golf"
SEED_PHASE_BUDGET="${SEED_PHASE_BUDGET:-1500}"
SEED_GRACE_SECONDS="${SEED_GRACE_SECONDS:-240}"
DOWNLOAD_RESERVE_SECONDS="${DOWNLOAD_RESERVE_SECONDS:-600}"
MIN_SECONDS_TO_START_SEED="${MIN_SECONDS_TO_START_SEED:-$((SEED_PHASE_BUDGET + DOWNLOAD_RESERVE_SECONDS))}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-240}"
SCP_RETRIES="${SCP_RETRIES:-3}"
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
POD_TERMINATED=0
SEEDS="${SEEDS:-42}"
BALANCE_FLOOR="${BALANCE_FLOOR:-50}"
BALANCE_CHECK_SECONDS="${BALANCE_CHECK_SECONDS:-30}"
BALANCE_WATCH_PID=""
POD_SSH_READY_TIMEOUT_SECONDS="${POD_SSH_READY_TIMEOUT_SECONDS:-180}"
POD_CREATE_RETRIES="${POD_CREATE_RETRIES:-3}"
POD_RETRY_SLEEP_SECONDS="${POD_RETRY_SLEEP_SECONDS:-15}"

# ── Local paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Auto-discover trainer path (local or project root)
if [[ -f "${SCRIPT_DIR:-.}/train_gpt.py" ]]; then
    TRAINER_PATH="${SCRIPT_DIR:-.}/train_gpt.py"
elif [[ -f "$(cd "${SCRIPT_DIR:-.}/../../.." 2>/dev/null && pwd)/train_gpt.py" ]]; then
    TRAINER_PATH="$(cd "${SCRIPT_DIR:-.}/../../.." && pwd)/train_gpt.py"
else
    # Fallback for scripts that don't define SCRIPT_DIR
    TRAINER_PATH="./train_gpt.py"
fi
LOCAL_ARTIFACTS_DIR="${LOCAL_ARTIFACTS_DIR:-${SCRIPT_DIR}/experiment_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_ARTIFACTS_DIR"
ORCH_LOG="${LOCAL_ARTIFACTS_DIR}/orchestrator.log"
DOWNLOAD_DIR="${LOCAL_ARTIFACTS_DIR}/downloads"
mkdir -p "$DOWNLOAD_DIR"

# ── Budget ────────────────────────────────────────────────────────────────────
T_START=""
T_BUDGET=5400    # 90 min: enough for 3 seeds × 10 min + data download + overhead

t_elapsed()   { [[ -z "$T_START" ]] && echo 0 || echo $(( $(date +%s) - T_START )); }
t_remaining() { [[ -z "$T_START" ]] && echo $T_BUDGET || echo $(( T_START + T_BUDGET - $(date +%s) )); }

log() {
    local e r mm ss
    e=$(t_elapsed); r=$(t_remaining)
    mm=$(( e / 60 )); ss=$(( e % 60 ))
    printf "[%02d:%02d | %4ds left] %s\n" "$mm" "$ss" "$r" "$*" | tee -a "$ORCH_LOG"
}

die() {
    log "FATAL: $*"
    [[ -n "${POD_ID:-}" ]] && terminate_pod "$POD_ID" || true
    exit 1
}

check_budget() {
    local label=$1 needed=$2
    if [[ $(t_remaining) -lt $needed ]]; then
        log "SKIP $label — only $(t_remaining)s left, need ${needed}s"
        return 1
    fi
    return 0
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

# ── RunPod API helper ─────────────────────────────────────────────────────────
gql() {
    curl -s --request POST \
        --header 'content-type: application/json' \
        --url "$RUNPOD_API" \
        --data "$(jq -n --arg q "$1" '{"query": $q}')"
}

list_live_paramgolf_pods() {
    gql 'query {
      myself {
        pods {
          id
          name
          desiredStatus
          runtime { uptimeInSeconds }
        }
      }
    }' | jq -r '
        .data.myself.pods[]?
        | select(.desiredStatus == "RUNNING")
        | select((.name // "") | startswith("paramgolf-"))
        | [.id, .name, (.runtime.uptimeInSeconds // 0 | tostring)]
        | @tsv
    '
}

list_live_pods() {
    gql 'query {
      myself {
        pods {
          id
          name
          desiredStatus
          runtime { uptimeInSeconds }
        }
      }
    }' | jq -r '
        .data.myself.pods[]?
        | select(.desiredStatus == "RUNNING")
        | [.id, (.name // ""), (.runtime.uptimeInSeconds // 0 | tostring)]
        | @tsv
    '
}

get_client_balance() {
    gql 'query { myself { clientBalance } }' | jq -r '.data.myself.clientBalance // empty'
}

gc_stale_pods() {
    local keep_name="${1:-}"
    local found=0
    while IFS=$'\t' read -r pid name uptime; do
        [[ -z "${pid:-}" ]] && continue
        [[ -n "$keep_name" && "$name" == "$keep_name" ]] && continue
        found=1
        log "GC: terminating stale live pod ${name} (${pid}, uptime=${uptime}s)"
        terminate_pod "$pid" || true
    done < <(list_live_paramgolf_pods)
    [[ $found -eq 1 ]] && sleep 5 || true
}

terminate_all_live_pods() {
    while IFS=$'\t' read -r pid name uptime; do
        [[ -z "${pid:-}" ]] && continue
        log "Balance guard: terminating live pod ${name:-unnamed} (${pid}, uptime=${uptime}s)"
        terminate_pod "$pid" || true
    done < <(list_live_pods)
}

balance_below_floor() {
    local balance="${1:-}"
    [[ -z "$balance" ]] && return 1
    awk -v b="$balance" -v floor="$BALANCE_FLOOR" 'BEGIN { exit !(b <= floor) }'
}

enforce_balance_floor() {
    local balance
    balance=$(get_client_balance)
    if balance_below_floor "$balance"; then
        log "Balance floor reached: clientBalance=${balance} <= ${BALANCE_FLOOR}"
        terminate_all_live_pods
        POD_TERMINATED=1
        exit 99
    fi
}

start_balance_watchdog() {
    local main_pid=$$
    (
        while true; do
            sleep "$BALANCE_CHECK_SECONDS"
            balance=""
            balance=$(get_client_balance)
            if balance_below_floor "$balance"; then
                log "Balance floor reached: clientBalance=${balance} <= ${BALANCE_FLOOR}"
                terminate_all_live_pods
                kill -TERM "$main_pid" 2>/dev/null || true
                break
            fi
        done
    ) &
    BALANCE_WATCH_PID=$!
}

# ── Pod lifecycle ─────────────────────────────────────────────────────────────
create_pod() {
    local pub_key
    pub_key=$(cat "$LOCAL_SSH_PUB") || die "Cannot read SSH public key: $LOCAL_SSH_PUB"

    log "Creating pod: ${GPU_COUNT}×${GPU_TYPE_ID} ${CLOUD_TYPE}..."
    log "  image: ${POD_IMAGE}"

    local mutation
    mutation=$(cat <<EOF
mutation {
  podFindAndDeployOnDemand(input: {
    gpuTypeId: "${GPU_TYPE_ID}"
    cloudType: ${CLOUD_TYPE}
    gpuCount: ${GPU_COUNT}
    volumeInGb: ${VOLUME_GB}
    containerDiskInGb: ${DISK_GB}
    minVcpuCount: 2
    minMemoryInGb: 15
    imageName: "${POD_IMAGE}"
    name: "${POD_NAME}"
    ports: "22/tcp"
    volumeMountPath: "/workspace"
    env: [{ key: "PUBLIC_KEY", value: "${pub_key}" }]
  }) {
    id name desiredStatus costPerHr
    runtime { ports { ip isIpPublic privatePort publicPort type } }
  }
}
EOF
)
    local result
    result=$(gql "$mutation")

    # Retry with SECURE if COMMUNITY fails
    if echo "$result" | jq -e '.errors' > /dev/null 2>&1; then
        log "COMMUNITY failed: $(echo "$result" | jq -r '.errors[].message' | head -2)"
        log "Retrying with SECURE cloud..."
        mutation="${mutation/cloudType: COMMUNITY/cloudType: SECURE}"
        result=$(gql "$mutation")
        echo "$result" | jq -e '.errors' > /dev/null 2>&1 \
            && die "Pod creation failed on both clouds: $(echo "$result" | jq -r '.errors[].message')"
    fi

    POD_ID=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.id')
    local cost_hr
    cost_hr=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.costPerHr // "unknown"')
    [[ -z "$POD_ID" || "$POD_ID" == "null" ]] && die "No pod ID in response: $result"

    echo "$POD_ID" > "${LOCAL_ARTIFACTS_DIR}/pod_id.txt"
    log "Pod created: ID=${POD_ID}  cost=\$${cost_hr}/hr"

    # Start budget clock at the first billable pod second and keep it across retries.
    if [[ -z "$T_START" ]]; then
        T_START=$(date +%s)
        log "BUDGET CLOCK STARTED — ${T_BUDGET}s total"
    else
        log "BUDGET CLOCK CONTINUES — ${T_BUDGET}s total, $(t_remaining)s left"
    fi
    POD_TERMINATED=0
}

wait_for_pod() {
    log "Waiting for pod SSH readiness..."
    local waited=0 MAX_WAIT="${POD_SSH_READY_TIMEOUT_SECONDS}"
    while [[ $waited -lt $MAX_WAIT ]]; do
        sleep 10; waited=$(( waited + 10 ))

        refresh_pod_endpoint || { log "  waiting... ${waited}s"; continue; }

        if ssh "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "echo SSH_OK" 2>/dev/null | grep -q "SSH_OK"; then
            log "Pod ready: root@${POD_HOST}:${POD_PORT}  T+$(t_elapsed)s"
            return 0
        fi
        log "  SSH not yet responding... ${waited}s"
    done
    log "Pod did not become SSH-ready after ${MAX_WAIT}s"
    return 1
}

terminate_pod() {
    local pid="${1:-${POD_ID:-}}"
    [[ -z "${pid:-}" || "$pid" == "null" ]] && return 0
    log "Terminating pod ${pid}..."
    gql "mutation { podTerminate(input: { podId: \"${pid}\" }) }" | jq -r '.data // .errors' | tee -a "$ORCH_LOG" || true
    POD_TERMINATED=1
}

create_ready_pod() {
    local attempt
    for (( attempt = 1; attempt <= POD_CREATE_RETRIES; attempt++ )); do
        log "Pod allocation attempt ${attempt}/${POD_CREATE_RETRIES}"
        POD_ID=""
        POD_HOST=""
        POD_PORT=""
        create_pod
        if wait_for_pod; then
            return 0
        fi
        log "Cold-start failure on pod ${POD_ID}; terminating and retrying"
        terminate_pod "$POD_ID" || true
        if [[ $attempt -lt $POD_CREATE_RETRIES ]]; then
            sleep "$POD_RETRY_SLEEP_SECONDS"
        fi
    done
    die "Failed to get an SSH-ready pod after ${POD_CREATE_RETRIES} attempts"
}

# ── SSH helpers ───────────────────────────────────────────────────────────────
SSH_BASE=(
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=15
    -o ServerAliveInterval=20
    -o ServerAliveCountMax=5
    -o BatchMode=yes
    -i "$LOCAL_SSH_KEY"
)

r()   { ssh  "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "$@"; }
ul()  { scp  "${SSH_BASE[@]}" -P "$POD_PORT" "$@" "root@${POD_HOST}:/workspace/"; }
dlr() { scp -r "${SSH_BASE[@]}" -P "$POD_PORT" "$@"; }

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "${BALANCE_WATCH_PID:-}" ]]; then
        kill "${BALANCE_WATCH_PID}" 2>/dev/null || true
    fi
    if [[ $exit_code -ne 0 && "${KEEP_POD_ON_EXIT}" != "1" && -n "${POD_ID:-}" && "${POD_TERMINATED}" -eq 0 ]]; then
        log "Cleanup: terminating pod after abnormal exit (code=${exit_code})"
        terminate_pod "$POD_ID" || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

refresh_pod_endpoint() {
    local result host port
    result=$(gql "query { pod(input: { podId: \"${POD_ID}\" }) { runtime { ports { ip isIpPublic privatePort publicPort } } } }")
    host=$(echo "$result" | jq -r '
        .data.pod.runtime.ports[]?
        | select(.privatePort == 22 and .isIpPublic == true) | .ip' | head -1)
    port=$(echo "$result" | jq -r '
        .data.pod.runtime.ports[]?
        | select(.privatePort == 22 and .isIpPublic == true) | .publicPort' | head -1)
    [[ -z "${host:-}" || "$host" == "null" || -z "${port:-}" || "$port" == "null" ]] && return 1
    POD_HOST="$host"
    POD_PORT="$port"
}

wait_for_ssh_ready() {
    local label=$1
    local max_wait=$2
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        refresh_pod_endpoint || true
        if [[ -n "${POD_HOST:-}" && -n "${POD_PORT:-}" ]] \
            && ssh "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "echo SSH_OK" 2>/dev/null | grep -q "SSH_OK"; then
            return 0
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    log "WARNING: ${label} SSH readiness check timed out after ${max_wait}s"
    return 1
}

download_from_pod() {
    local remote_path=$1
    local local_path=$2
    local recursive=${3:-0}
    local optional=${4:-0}
    local attempt
    for (( attempt = 1; attempt <= SCP_RETRIES; attempt++ )); do
        refresh_pod_endpoint >/dev/null 2>&1 || true
        if [[ "$recursive" == "1" ]]; then
            scp -r "${SSH_BASE[@]}" -P "$POD_PORT" "root@${POD_HOST}:${remote_path}" "$local_path" >/dev/null 2>&1 && return 0
        else
            scp "${SSH_BASE[@]}" -P "$POD_PORT" "root@${POD_HOST}:${remote_path}" "$local_path" >/dev/null 2>&1 && return 0
        fi
        sleep 5
    done
    [[ "$optional" == "1" ]] && return 1
    log "WARNING: failed to download ${remote_path} after ${SCP_RETRIES} attempts"
    return 1
}

sync_seed_artifacts() {
    local label=$1
    local run_id=${2:-}
    wait_for_ssh_ready "${label} artifact sync" 120 || return 1
    download_from_pod "/workspace/logs/${label}_run.log" "${LOCAL_ARTIFACTS_DIR}/${label}.log" 0 1 || true
    [[ -n "$run_id" ]] || return 0
    download_from_pod "/workspace/logs/${run_id}.log" "${DOWNLOAD_DIR}/${run_id}.log" 0 1 || true
    download_from_pod "/workspace/logs/${run_id}_model.ternary.ptz" "${DOWNLOAD_DIR}/${run_id}_model.ternary.ptz" 0 1 || true
    download_from_pod "/workspace/logs/${run_id}_submission.json" "${DOWNLOAD_DIR}/${run_id}_submission.json" 0 1 || true
}

# ── Training seed runner ──────────────────────────────────────────────────────
run_seed() {
    local SEED=$1
    local PHASE_BUDGET=$2
    local LABEL="seed${SEED}"
    local SESSION="pg_${LABEL}"
    local REMOTE_LOG="/workspace/logs/${LABEL}_run.log"
    local LOCAL_LOG="${LOCAL_ARTIFACTS_DIR}/${LABEL}.log"

    log "--- ${LABEL}: starting (phase_budget=${PHASE_BUDGET}s, $(t_remaining)s overall left) ---"
    check_budget "$LABEL" "$MIN_SECONDS_TO_START_SEED" || { log "SKIP $LABEL — not enough time"; return 0; }

    r "tmux kill-session -t ${SESSION} 2>/dev/null || true; mkdir -p /workspace/logs"
    r "cd /workspace && tmux new-session -d -s ${SESSION} \
        \"SEED=${SEED} bash run_h100_skc_competition.sh 2>&1 | tee ${REMOTE_LOG}\""
    log "$LABEL launched in tmux '${SESSION}'"

    local PHASE_DEADLINE=$(( $(date +%s) + PHASE_BUDGET ))
    local HARD_DEADLINE=$(( PHASE_DEADLINE + SEED_GRACE_SECONDS ))
    local PREV_LINE="" COMPLETED=false GRACE_LOGGED=false STOP_REASON=""

    while true; do
        sleep 30

        if [[ $(date +%s) -ge $PHASE_DEADLINE && "$GRACE_LOGGED" == "false" ]]; then
            log "$LABEL: phase budget reached — allowing ${SEED_GRACE_SECONDS}s grace for serialization/eval"
            GRACE_LOGGED=true
        fi
        if [[ $(date +%s) -ge $HARD_DEADLINE ]]; then
            STOP_REASON="phase budget exhausted"
            break
        fi
        if [[ $(t_remaining) -le $DOWNLOAD_RESERVE_SECONDS ]]; then
            STOP_REASON="global download reserve reached"
            break
        fi

        download_from_pod "$REMOTE_LOG" "$LOCAL_LOG" 0 1 || true

        if grep -q "=== DONE ===" "$LOCAL_LOG" 2>/dev/null; then
            COMPLETED=true
            log "$LABEL COMPLETED ✓  T+$(t_elapsed)s"
            break
        fi

        local LINE
        LINE=$(grep -E "^step:|val_bpb:|final_sliding|final_ternary|hessian_ternary|gptq|artifact|ngram_cache" \
                   "$LOCAL_LOG" 2>/dev/null | tail -1)
        if [[ -n "$LINE" && "$LINE" != "$PREV_LINE" ]]; then
            log "$LABEL » $LINE"
            PREV_LINE="$LINE"
        fi
    done

    if [[ "$COMPLETED" != "true" ]]; then
        log "$LABEL: ${STOP_REASON:-stopping early}"
        r "tmux has-session -t ${SESSION} 2>/dev/null && tmux send-keys -t ${SESSION} C-c" || true
        sleep 15
    fi

    r "tmux kill-session -t ${SESSION} 2>/dev/null || true" || true

    # Final log sync
    download_from_pod "$REMOTE_LOG" "$LOCAL_LOG" 0 1 || true

    # Tag per-seed artifacts on remote before next seed overwrites
    local RUN_ID
    RUN_ID=$(grep "RUN_ID" "$LOCAL_LOG" 2>/dev/null | head -1 | grep -oE "skc_h100_s[0-9]+_[0-9_]+" | head -1)
    if [[ -n "$RUN_ID" ]]; then
        r "cp /workspace/logs/${RUN_ID}_model.ternary.ptz /workspace/logs/${RUN_ID}_tagged.ternary.ptz 2>/dev/null || \
           cp /workspace/logs/mlx_reasoner_model.ternary.ptz /workspace/logs/${RUN_ID}_model.ternary.ptz 2>/dev/null || true"
        log "$LABEL tagged artifacts as RUN_ID=${RUN_ID}"
    else
        r "cp /workspace/logs/mlx_reasoner_model.ternary.ptz /workspace/logs/${LABEL}_model.ternary.ptz 2>/dev/null || true"
        log "$LABEL WARNING: no RUN_ID parsed — tagged as $LABEL"
    fi
    sync_seed_artifacts "$LABEL" "$RUN_ID" || true

    local RESULT
    RESULT=$(grep -E "final_sliding val_bpb:|ngram_cache val_bpb:|val_bpb:" \
                 "$LOCAL_LOG" 2>/dev/null | tail -1)
    log "$LABEL result: ${RESULT:-no BPB line found yet}"
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
log "========================================================"
log "PARAMETER GOLF — H100 EXPERIMENT (follow-the-leader)"
log "Stack    : hardware-aware SKC arch (runner chooses profile) vocab=1024 sp1024"
log "Budget   : ${T_BUDGET}s (59:59)"
log "Seeds    : ${SEEDS}"
log "Cloud    : ${CLOUD_TYPE}"
log "SSH key  : ${LOCAL_SSH_KEY}"
log "Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "========================================================"

# Pre-flight checks
require_cmd curl
require_cmd jq
require_cmd ssh
require_cmd scp
[[ -f "$LOCAL_SSH_KEY" ]]    || die "SSH private key not found: $LOCAL_SSH_KEY"
[[ -f "$LOCAL_SSH_PUB" ]]    || die "SSH public key not found: $LOCAL_SSH_PUB"
[[ -f "${TRAINER_PATH}" ]] || die "train_gpt.py not found in $SCRIPT_DIR"
[[ -f "${SCRIPT_DIR}/run_h100_skc_competition.sh" ]] || die "run_h100_skc_competition.sh not found in $SCRIPT_DIR"
gc_stale_pods "$POD_NAME"
enforce_balance_floor

# ── 1. Create pod ─────────────────────────────────────────────────────────────
POD_ID=""
POD_HOST="" POD_PORT=""
create_ready_pod
start_balance_watchdog

# ── 3. GPU + PyTorch verification ─────────────────────────────────────────────
log "=== PHASE: Verification ==="
r "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" | tee -a "$ORCH_LOG" \
    || die "nvidia-smi failed"
GPU_COUNT_ACTUAL=$(r "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
[[ "$GPU_COUNT_ACTUAL" -eq 8 ]] || log "WARNING: expected 8 GPUs, got $GPU_COUNT_ACTUAL"
r "python3 -c 'import torch; print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"gpus\",torch.cuda.device_count())'" \
    | tee -a "$ORCH_LOG" || die "PyTorch unavailable"

# ── 4. Install dependencies ───────────────────────────────────────────────────
log "=== PHASE: Install dependencies ==="

log "FlashAttention: SKC architecture uses WHT spectral mixing, NOT attention — FA not needed."
log "  (fallback to torch.sdpa exists in code for non-SKC architectures but won't be used)"

log "Installing sentencepiece + zstandard..."
r "pip install -q sentencepiece zstandard && echo 'pip OK'" | tee -a "$ORCH_LOG" \
    || die "pip install failed"

log "Installing tmux..."
r "which tmux > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -q tmux)" \
    | tee -a "$ORCH_LOG" || true

log "Installing huggingface-hub..."
r "pip install -q huggingface-hub && echo 'hf hub OK'" | tee -a "$ORCH_LOG" || die "hf hub install failed"

# ── 5. Download data ──────────────────────────────────────────────────────────
log "=== PHASE: Download sp1024 data (${DATA_SHARDS} train shards + val) ==="
check_budget "data_download" 600 || die "Not enough time to download data"

r "mkdir -p /workspace/data/datasets/fineweb10B_sp1024 /workspace/data/tokenizers"

# Download val shard first (small, needed for smoke test)
log "Downloading val shard..."
r "python3 -c \"
from huggingface_hub import hf_hub_download
import shutil, os
dest = '/workspace/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'
if os.path.exists(dest):
    print('val shard already exists')
else:
    p = hf_hub_download(repo_id='${HF_REPO}', filename='fineweb_val_000000.bin',
        subfolder='datasets/datasets/fineweb10B_sp1024', repo_type='dataset')
    import pathlib; src = pathlib.Path(p).resolve()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try: os.link(src, dest)
    except: shutil.copy2(src, dest)
    print('val shard downloaded')
\"" | tee -a "$ORCH_LOG" || die "Val shard download failed"

# Download tokenizer
log "Downloading tokenizer..."
r "python3 -c \"
from huggingface_hub import hf_hub_download
import shutil, os
dest = '/workspace/data/tokenizers/fineweb_1024_bpe.model'
if os.path.exists(dest):
    print('tokenizer already exists')
else:
    p = hf_hub_download(repo_id='${HF_REPO}', filename='fineweb_1024_bpe.model',
        subfolder='datasets/tokenizers', repo_type='dataset')
    import pathlib; src = pathlib.Path(p).resolve()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try: os.link(src, dest)
    except: shutil.copy2(src, dest)
    print('tokenizer downloaded')
\"" | tee -a "$ORCH_LOG" || die "Tokenizer download failed"

# Download train shards in parallel (background jobs, max 8 concurrent)
log "Downloading ${DATA_SHARDS} train shards in parallel..."
r "python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil, os, pathlib, sys, time

repo = '${HF_REPO}'
dest_dir = '/workspace/data/datasets/fineweb10B_sp1024'
os.makedirs(dest_dir, exist_ok=True)
shards = [f'fineweb_train_{i:06d}.bin' for i in range(${DATA_SHARDS})]

def download_shard(fname, retries=3):
    dest = os.path.join(dest_dir, fname)
    if os.path.exists(dest):
        return f'{fname}: already exists'
    for attempt in range(retries):
        try:
            p = hf_hub_download(repo_id=repo, filename=fname,
                subfolder='datasets/datasets/fineweb10B_sp1024', repo_type='dataset')
            src = pathlib.Path(p).resolve()
            try: os.link(src, dest)
            except: shutil.copy2(src, dest)
            return f'{fname}: done'
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f'{fname}: FAILED ({e})'

completed = 0
failed = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(download_shard, s): s for s in shards}
    for f in as_completed(futs):
        result = f.result()
        completed += 1
        if 'FAILED' in result:
            failed += 1
            print(f'  WARNING: {result}', flush=True)
        if completed % 10 == 0 or completed == len(shards):
            print(f'  {completed}/{len(shards)} shards done ({failed} failed)', flush=True)

actual = len([f for f in os.listdir(dest_dir) if f.startswith('fineweb_train_')])
print(f'Download complete: {actual}/{len(shards)} shards available ({failed} failed)')
if actual < 10:
    sys.exit(1)  # Need at least 10 shards
PYEOF" | tee -a "$ORCH_LOG" || log "WARNING: Some train shards may have failed"

# Verify
TRAIN_COUNT=$(r "ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l")
VAL_COUNT=$(r "ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l")
log "Data verified: train_shards=${TRAIN_COUNT}  val_shards=${VAL_COUNT}  T+$(t_elapsed)s"
[[ "$TRAIN_COUNT" -gt 0 ]] || die "No train shards found after download"
[[ "$VAL_COUNT"   -gt 0 ]] || die "No val shards found after download"

# ── 6. Upload code ────────────────────────────────────────────────────────────
log "=== PHASE: Upload code ==="
ul "${TRAINER_PATH}" "${SCRIPT_DIR}/run_h100_skc_competition.sh" \
    || die "Code upload failed"
r "chmod +x /workspace/run_h100_skc_competition.sh && mkdir -p /workspace/logs"
log "Code uploaded  T+$(t_elapsed)s"

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
log "=== PHASE: Smoke test (FAST_SMOKE=1) ==="
check_budget "smoke_test" 300 || die "Not enough time for smoke test"

SMOKE_LOG="${LOCAL_ARTIFACTS_DIR}/smoke_test.log"
# Smoke test: keep DDP on, but disable slow export/eval features so success/failure is unambiguous.
r "cd /workspace && FAST_SMOKE=1 MAX_WALLCLOCK_SECONDS=45 COMPILE_MODE=none OMP_NUM_THREADS=1 timeout ${SMOKE_TIMEOUT_SECONDS} \
    bash run_h100_skc_competition.sh 2>&1" \
    | tee "$SMOKE_LOG" || log "WARNING: smoke test non-zero exit"

# Validate smoke
SMOKE_NAN=$(grep -cE "val_bpb:nan|loss:nan" "$SMOKE_LOG" 2>/dev/null || true)
SMOKE_ERR=$(grep -cE "^Traceback|CUDA error|RuntimeError|AssertionError|ImportError" "$SMOKE_LOG" 2>/dev/null || true)
SMOKE_OK=$(grep -cE "val_bpb:[0-9]|step:[0-9]" "$SMOKE_LOG" 2>/dev/null || true)

[[ "$SMOKE_NAN" -gt 0 ]] && die "Smoke test: NaN detected. See smoke_test.log"
[[ "$SMOKE_ERR" -gt 0 ]] && die "Smoke test: hard errors. See smoke_test.log"
[[ "$SMOKE_OK"  -eq 0 ]] && die "Smoke test: no training output. See smoke_test.log"

SMOKE_BPB=$(grep "val_bpb:" "$SMOKE_LOG" | tail -1 | grep -oE "val_bpb:[0-9.]+" | cut -d: -f2)
log "Smoke test PASSED ✓  bpb=${SMOKE_BPB:-n/a}  T+$(t_elapsed)s"

# ── 8. Configured seeds ───────────────────────────────────────────────────────
for seed in $SEEDS; do
    enforce_balance_floor
    log "=== PHASE: Seed ${seed} (phase_budget=${SEED_PHASE_BUDGET}s) ==="
    run_seed "$seed" "$SEED_PHASE_BUDGET"
done

# ── 10. Download all artifacts ────────────────────────────────────────────────
log "=== PHASE: Download all artifacts ($(t_remaining)s remaining) ==="
wait_for_ssh_ready "final artifact download" 180 || true

log "Downloading /workspace/logs/ ..."
download_from_pod "/workspace/logs/." "$DOWNLOAD_DIR/" 1 1 \
    && log "  /workspace/logs synced" \
    || log "WARNING: partial log download"

log "Downloading artifacts from /workspace/ and /workspace/logs/ ..."
for f in final_model.ternary.ptz submission.json; do
    download_from_pod "/workspace/${f}" "${DOWNLOAD_DIR}/${f}" 0 1 && log "  ${f} downloaded" || true
done
for f in mlx_reasoner_model.ternary.ptz; do
    download_from_pod "/workspace/logs/${f}" "${DOWNLOAD_DIR}/${f}" 0 1 && log "  ${f} downloaded" || true
done

# ── 11. Terminate pod ─────────────────────────────────────────────────────────
log "=== Terminating pod ${POD_ID} ==="
terminate_pod "$POD_ID"

# ── Final summary ─────────────────────────────────────────────────────────────
log "========================================================"
log "EXPERIMENT COMPLETE — $(t_elapsed)s / ${T_BUDGET}s used"
log "========================================================"
log ""
log "All artifacts at: ${LOCAL_ARTIFACTS_DIR}"
log ""
log "Files:"
find "$LOCAL_ARTIFACTS_DIR" \( -name "*.ptz" -o -name "*.pt" -o -name "*.json" -o -name "*.log" \) \
    | sort | while read -r f; do
    SIZE=$(ls -lh "$f" 2>/dev/null | awk '{print $5}')
    log "  [${SIZE:-?}]  $f"
done

log ""
log "BPB results:"
for f in "${LOCAL_ARTIFACTS_DIR}"/seed*.log "${DOWNLOAD_DIR}"/*.log; do
    [[ -f "$f" ]] || continue
    RESULT=$(grep -E "ngram_cache val_bpb:|final_sliding val_bpb:|final_ternary_roundtrip val_bpb:|val_bpb:" \
                 "$f" 2>/dev/null | tail -1)
    [[ -n "$RESULT" ]] && log "  $(basename "$f"): $RESULT"
done

log ""
log "Submit the .ternary.ptz with the lowest ngram_cache / final_sliding val_bpb"
log "Full log: $ORCH_LOG"
