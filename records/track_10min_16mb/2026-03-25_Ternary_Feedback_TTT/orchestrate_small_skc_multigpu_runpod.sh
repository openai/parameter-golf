#!/usr/bin/env bash
# ============================================================================
# Cheap multi-GPU RunPod orchestrator for the small SKC proxy run.
# Chooses the lowest-cost viable 2-GPU community pod live, with fallbacks.
# ============================================================================
set -uo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"

LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="paramgolf-small-$(date +%Y%m%d-%H%M%S)"
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
GPU_COUNT="${GPU_COUNT:-2}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-24}"
VOLUME_GB="${VOLUME_GB:-20}"
DISK_GB="${DISK_GB:-20}"
MIN_VCPU_COUNT="${MIN_VCPU_COUNT:-1}"
MIN_SYSTEM_MEMORY_GB="${MIN_SYSTEM_MEMORY_GB:-8}"
DATA_SHARDS="${DATA_SHARDS:-12}"
HF_REPO="willdepueoai/parameter-golf"
BALANCE_FLOOR="30"
BALANCE_CHECK_SECONDS="${BALANCE_CHECK_SECONDS:-30}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-180}"
LATENT_CHECKPOINT_LOCAL_PATH="${LATENT_CHECKPOINT_LOCAL_PATH:-}"
SCP_RETRIES="${SCP_RETRIES:-3}"
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
POD_TERMINATED=0
BALANCE_WATCH_PID=""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ARTIFACTS_DIR="${LOCAL_ARTIFACTS_DIR:-${SCRIPT_DIR}/small_skc_multigpu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_ARTIFACTS_DIR"
DOWNLOAD_DIR="${LOCAL_ARTIFACTS_DIR}/downloads"
mkdir -p "$DOWNLOAD_DIR"
ORCH_LOG="${LOCAL_ARTIFACTS_DIR}/orchestrator.log"

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$ORCH_LOG"; }
die() {
    log "FATAL: $*"
    [[ -n "${POD_ID:-}" ]] && terminate_pod "$POD_ID" || true
    exit 1
}
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

REMOTE_ENV_KEYS=(
    BIGRAM_HASH_DIM
    TRAIN_BATCH_TOKENS
    VAL_BATCH_SIZE
    MATRIX_LR
    SCALAR_LR
    SELF_DISTILL_KL_WEIGHT
    LAWA_ENABLED
    SWA_ENABLED
    AVERAGE_TERNARY_PARAMS
    TURBO_QUANT_TRAIN
    TURBO_QUANT_EXPORT
    HESSIAN_TERNARY_GPTQ
    GPTQ_LITE_ENABLED
    SELECTIVE_PRUNING
    EXPORT_ALIGNED_TRAIN
    EXPORT_ALIGNED_TRAIN_START_FRACTION
    TERNARY_THRESHOLD_SEARCH
    TERNARY_THRESHOLD_LOW
    TERNARY_THRESHOLD_HIGH
    TERNARY_THRESHOLD_STEPS
    TERNARY_SCALE_SEARCH
    TERNARY_SCALE_MULT_LOW
    TERNARY_SCALE_MULT_HIGH
    TERNARY_SCALE_MULT_STEPS
    NGRAM_CACHE_ENABLED
    EXPORT_ONLY
    LATENT_CHECKPOINT_PATH
    SEED
    NUM_LAYERS
    MODEL_DIM
    KOOPMAN_RANK
    KOOPMAN_MIXER_RANK
    MOE_ENABLED
    MOE_NUM_EXPERTS
    MOE_TOP_K
    BIGRAM_HASH_BUCKETS
    ARCHITECTURE
    SKC_NUM_CAPSULES
    SKC_CAPSULE_DIM
    SKC_CONV_KERNEL
    SKC_BLOCK_SIZE
    FEEDBACK_ENABLED
    CAPSULE_ENABLED
    DEQ_FEEDBACK
    DEQ_MAX_ITER
    VRL_ENABLED
    KOOPMAN_ENABLED
    KOOPMAN_SPECULATOR_ENABLED
    TTT_ENABLED
    EMA_ENABLED
    WEIGHT_SHARING
    INSIDE_OUT_TRAINING
    TKO_ENABLED
    MOE_START_FRACTION
    BIGRAM_HASH_ENABLED
    ENGRAM_NUM_HEADS
    ENGRAM_NUM_ORDERS
    ENGRAM_INJECT_LAYER
    LN_SCALE_DAMPING
    SMEARGATE_ENABLED
    MAX_WALLCLOCK_SECONDS
    PARTIAL_ROPE_DIMS
    CURRICULUM_ENABLED
)

build_remote_env_prefix() {
    local prefix=""
    local key value escaped
    for key in "${REMOTE_ENV_KEYS[@]}"; do
        if [[ -n "${!key+x}" ]]; then
            value="${!key}"
            escaped=$(printf '%q' "$value")
            prefix+="${key}=${escaped} "
        fi
    done
    printf "%s" "$prefix"
}

gql() {
    curl -s --request POST \
        --header "content-type: application/json" \
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

balance_below_floor() {
    local balance="${1:-}"
    [[ -z "$balance" ]] && return 1
    awk -v b="$balance" -v floor="$BALANCE_FLOOR" 'BEGIN { exit !(b <= floor) }'
}

terminate_pod() {
    local pid="${1:-${POD_ID:-}}"
    [[ -z "${pid:-}" || "$pid" == "null" ]] && return 0
    if [[ "${KEEP_POD_ON_EXIT:-0}" -eq 1 ]]; then
        log "KEEP_POD_ON_EXIT=1, skipping termination of pod ${pid}."
        return 0
    fi
    log "Terminating pod ${pid}..."
    gql "mutation { podTerminate(input: { podId: \"${pid}\" }) }" | jq -r '.data // .errors' | tee -a "$ORCH_LOG" || true
    POD_TERMINATED=1
}

terminate_all_live_pods() {
    while IFS=$'\t' read -r pid name uptime; do
        [[ -z "${pid:-}" ]] && continue
        log "Balance guard: terminating live pod ${name:-unnamed} (${pid}, uptime=${uptime}s)"
        terminate_pod "$pid" || true
    done < <(list_live_pods)
}

gc_stale_pods() {
    local keep_name="${1:-}"
    while IFS=$'\t' read -r pid name uptime; do
        [[ -z "${pid:-}" ]] && continue
        [[ -n "$keep_name" && "$name" == "$keep_name" ]] && continue
        log "GC: terminating stale live pod ${name} (${pid}, uptime=${uptime}s)"
        terminate_pod "$pid" || true
    done < <(list_live_paramgolf_pods)
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

pick_gpu_candidates() {
    gql "query {
      gpuTypes {
        id
        displayName
        memoryInGb
        lowestPrice(input: { gpuCount: ${GPU_COUNT}, secureCloud: false }) {
          uninterruptablePrice
          stockStatus
        }
      }
    }" | jq -r --argjson min_mem "$MIN_GPU_MEMORY_GB" '
        .data.gpuTypes[]?
        | select(.lowestPrice.uninterruptablePrice != null)
        | select((.memoryInGb // 0) >= $min_mem)
        | select(((.displayName + " " + .id) | test("V100|P100|T4|K80|M60|A2|A16|MI|Radeon|Instinct|Blackwell|PRO 6000"; "i")) | not)
        | [.lowestPrice.uninterruptablePrice, .id, .lowestPrice.stockStatus, (.memoryInGb | tostring), .displayName]
        | @tsv
    ' | sort -n
}

pick_secure_gpu_candidates() {
    gql "query {
      gpuTypes {
        id
        displayName
        memoryInGb
        lowestPrice(input: { gpuCount: ${GPU_COUNT}, secureCloud: true }) {
          uninterruptablePrice
          stockStatus
        }
      }
    }" | jq -r --argjson min_mem "$MIN_GPU_MEMORY_GB" '
        .data.gpuTypes[]?
        | select(.lowestPrice.uninterruptablePrice != null)
        | select((.memoryInGb // 0) >= $min_mem)
        | select(((.displayName + " " + .id) | test("V100|P100|T4|K80|M60|A2|A16|MI|Radeon|Instinct|Blackwell|PRO 6000"; "i")) | not)
        | [.lowestPrice.uninterruptablePrice, .id, .lowestPrice.stockStatus, (.memoryInGb | tostring), .displayName]
        | @tsv
    ' | sort -n
}

try_create_pod() {
    local gpu_type="$1"
    local cloud="$2"
    local pub_key
    pub_key=$(cat "$LOCAL_SSH_PUB") || die "Cannot read SSH public key: $LOCAL_SSH_PUB"
    log "Trying ${GPU_COUNT}x${gpu_type} on ${cloud}..."

    local mutation result
    mutation=$(cat <<EOF
mutation {
  podFindAndDeployOnDemand(input: {
    gpuTypeId: "${gpu_type}"
    cloudType: ${cloud}
    gpuCount: ${GPU_COUNT}
    volumeInGb: ${VOLUME_GB}
    containerDiskInGb: ${DISK_GB}
    minVcpuCount: ${MIN_VCPU_COUNT}
    minMemoryInGb: ${MIN_SYSTEM_MEMORY_GB}
    imageName: "${POD_IMAGE}"
    name: "${POD_NAME}"
    ports: "22/tcp"
    volumeMountPath: "/workspace"
    env: [{ key: "PUBLIC_KEY", value: "${pub_key}" }]
  }) {
    id
    costPerHr
  }
}
EOF
)
    result=$(gql "$mutation")
    if echo "$result" | jq -e '.errors' >/dev/null 2>&1; then
        log "  failed: $(echo "$result" | jq -r '.errors[].message' | head -1)"
        return 1
    fi
    POD_ID=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
    [[ -z "$POD_ID" ]] && return 1
    local cost
    cost=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.costPerHr // "unknown"')
    log "  success: pod=${POD_ID} cost=\$${cost}/hr"
    echo "${gpu_type}" > "${LOCAL_ARTIFACTS_DIR}/gpu_type.txt"
    echo "${cloud}" > "${LOCAL_ARTIFACTS_DIR}/cloud_type.txt"
    return 0
}

create_pod() {
    if [[ -n "${POD_ID:-}" && "$POD_ID" != "null" ]]; then
        log "Re-using existing pod: ${POD_ID}"
        # We need to refresh POD_HOST and POD_PORT for the existing pod
        refresh_pod_endpoint
        return 0
    fi
    local candidates secure_candidates
    candidates=$(pick_gpu_candidates)
    [[ -n "$candidates" ]] || die "No viable ${GPU_COUNT}-GPU community candidates found"
    printf "%s\n" "$candidates" | tee -a "$ORCH_LOG" > "${LOCAL_ARTIFACTS_DIR}/gpu_candidates.tsv"
    secure_candidates=$(pick_secure_gpu_candidates)
    [[ -n "$secure_candidates" ]] || die "No viable ${GPU_COUNT}-GPU secure candidates found"
    printf "%s\n" "$secure_candidates" | tee -a "$ORCH_LOG" > "${LOCAL_ARTIFACTS_DIR}/gpu_secure_candidates.tsv"

    local phase cloud pool key
    local tried=$'\n'
    for phase in high_or_medium any; do
        for cloud in SECURE COMMUNITY; do
            if [[ "$cloud" == "SECURE" ]]; then
                pool="$secure_candidates"
            else
                pool="$candidates"
            fi
            while IFS=$'\t' read -r price gpu_id stock mem display_name; do
                [[ -z "${gpu_id:-}" ]] && continue
                if [[ "$phase" == "high_or_medium" && "$stock" == "Low" ]]; then
                    continue
                fi
                key="${cloud}|${gpu_id}"
                if [[ "$tried" == *$'\n'"$key"$'\n'* ]]; then
                    continue
                fi
                tried+="${key}"$'\n'
                if [[ "$cloud" == "SECURE" ]]; then
                    log "Secure candidate: ${display_name} (${gpu_id})  ${GPU_COUNT}x  mem=${mem}GB  stock=${stock}  price=\$${price}/hr"
                else
                    log "Candidate: ${display_name} (${gpu_id})  ${GPU_COUNT}x  mem=${mem}GB  stock=${stock}  price=\$${price}/hr"
                fi
                try_create_pod "$gpu_id" "$cloud" && return 0
            done <<< "$pool"
        done
    done
    die "Could not create a cheap multi-GPU pod"
}

SSH_BASE=(
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=15
    -o ServerAliveInterval=20
    -o ServerAliveCountMax=5
    -o BatchMode=yes
    -i "$LOCAL_SSH_KEY"
)

r()   { ssh "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "$@"; }
ul()  { scp "${SSH_BASE[@]}" -P "$POD_PORT" "$@" "root@${POD_HOST}:/workspace/"; }

r_retry() {
    local attempt
    for (( attempt = 1; attempt <= 3; attempt++ )); do
        refresh_pod_endpoint >/dev/null 2>&1 || true
        if r "$@"; then
            return 0
        fi
        sleep 5
    done
    return 1
}

ul_retry() {
    local attempt
    for (( attempt = 1; attempt <= 3; attempt++ )); do
        refresh_pod_endpoint >/dev/null 2>&1 || true
        if ul "$@"; then
            return 0
        fi
        sleep 5
    done
    return 1
}

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
    host=$(echo "$result" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort == 22 and .isIpPublic == true) | .ip' | head -1)
    port=$(echo "$result" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort == 22 and .isIpPublic == true) | .publicPort' | head -1)
    [[ -z "${host:-}" || "$host" == "null" || -z "${port:-}" || "$port" == "null" ]] && return 1
    POD_HOST="$host"
    POD_PORT="$port"
}

wait_for_pod() {
    local waited=0
    log "Waiting for pod SSH readiness..."
    while [[ $waited -lt 600 ]]; do
        sleep 10
        waited=$(( waited + 10 ))
        refresh_pod_endpoint || continue
        if ssh "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "echo SSH_OK" 2>/dev/null | grep -q "SSH_OK"; then
            log "Pod ready: root@${POD_HOST}:${POD_PORT}"
            return 0
        fi
    done
    die "Pod not SSH-ready after 600s"
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
        sleep 3
    done
    [[ "$optional" == "1" ]] && return 1
    log "WARNING: failed to download ${remote_path}"
    return 1
}

wait_for_log_token() {
    local remote_log=$1
    local pattern=$2
    local timeout=${3:-1200}
    local waited=0
    local tmp="${LOCAL_ARTIFACTS_DIR}/poll.log"
    while [[ $waited -lt $timeout ]]; do
        sleep 10
        waited=$(( waited + 10 ))
        download_from_pod "$remote_log" "$tmp" 0 1 || true
        if [[ -f "$tmp" ]] && grep -q "$pattern" "$tmp"; then
            return 0
        fi
        if [[ -f "$tmp" ]]; then
            line=$(grep -E "^step:|val_bpb:|final_sliding|ngram_cache|artifact:" "$tmp" | tail -1)
            [[ -n "${line:-}" ]] && log "progress » ${line}"
        fi
    done
    return 1
}

write_remote_file() {
    local remote_path=$1
    local content=$2
    local enc_path enc_content
    enc_path=$(printf '%s' "$remote_path" | base64 | tr -d '\n')
    enc_content=$(printf '%s' "$content" | base64 | tr -d '\n')
    r_retry "python3 - <<'PYEOF'
from base64 import b64decode
from pathlib import Path
path = Path(b64decode('${enc_path}').decode('utf-8'))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(b64decode('${enc_content}').decode('utf-8'), encoding='utf-8')
path.chmod(0o755)
print(f'wrote:{path}')
PYEOF"
}

launch_remote_job() {
    local session=$1
    local script_path=$2
    local log_path=$3
    r_retry "tmux kill-session -t ${session} 2>/dev/null || true; rm -f ${log_path}; tmux new-session -d -s ${session} 'bash ${script_path} > ${log_path} 2>&1'"
}

download_run_artifacts() {
    local run_id=$1
    download_from_pod "/workspace/logs/${run_id}.log" "${DOWNLOAD_DIR}/${run_id}.log" 0 1 || true
    download_from_pod "/workspace/logs/${run_id}_model.ternary.ptz" "${DOWNLOAD_DIR}/${run_id}_model.ternary.ptz" 0 1 || true
    download_from_pod "/workspace/logs/${run_id}_submission.json" "${DOWNLOAD_DIR}/${run_id}_submission.json" 0 1 || true
    download_from_pod "/workspace/logs/${run_id}_pre_export_state.pt" "${DOWNLOAD_DIR}/${run_id}_pre_export_state.pt" 0 1 || true
}

log "========================================================"
log "Cheap Multi-GPU Small SKC Orchestrator"
log "Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "GPU count: ${GPU_COUNT}"
log "Min mem  : ${MIN_GPU_MEMORY_GB}GB"
log "Data     : ${DATA_SHARDS} train shards"
log "========================================================"

require_cmd curl
require_cmd jq
require_cmd ssh
require_cmd scp
require_cmd python3
[[ -f "$LOCAL_SSH_KEY" ]] || die "SSH private key not found: $LOCAL_SSH_KEY"
[[ -f "$LOCAL_SSH_PUB" ]] || die "SSH public key not found: $LOCAL_SSH_PUB"
[[ -f "${SCRIPT_DIR}/train_gpt.py" ]] || die "train_gpt.py not found in $SCRIPT_DIR"
[[ -f "${SCRIPT_DIR}/run_small_skc_2gpu.sh" ]] || die "run_small_skc_2gpu.sh not found in $SCRIPT_DIR"

gc_stale_pods "$POD_NAME"
balance=$(get_client_balance)
if balance_below_floor "$balance"; then
    die "Balance floor already reached: clientBalance=${balance} <= ${BALANCE_FLOOR}"
fi

create_pod
echo "$POD_ID" > "${LOCAL_ARTIFACTS_DIR}/pod_id.txt"
start_balance_watchdog
wait_for_pod

log "Verifying GPU and runtime..."
r_retry "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" | tee -a "$ORCH_LOG" || die "nvidia-smi failed"
r_retry "python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpus\", torch.cuda.device_count())'" | tee -a "$ORCH_LOG" || die "PyTorch unavailable"
GPU_COUNT_ACTUAL=$(r_retry "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
[[ "$GPU_COUNT_ACTUAL" -eq "$GPU_COUNT" ]] || log "WARNING: expected ${GPU_COUNT} GPUs, got ${GPU_COUNT_ACTUAL}"

log "Installing tmux and preparing logs..."
r_retry "apt-get update -qq && apt-get install -y -q tmux && mkdir -p /workspace/logs" || die "Failed to install tmux"

log "Installing dependencies..."
INSTALL_SCRIPT_CONTENT=$(cat <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if ! which tmux >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -q tmux
fi
pip install -q sentencepiece zstandard huggingface-hub
echo deps_ok
echo "=== INSTALL DONE ==="
EOF
)
write_remote_file "/workspace/install_setup.sh" "$INSTALL_SCRIPT_CONTENT" | tee -a "$ORCH_LOG" || die "Failed to write install script"
launch_remote_job "small_setup_install" "/workspace/install_setup.sh" "/workspace/logs/install_setup.log" || die "Failed to launch install job"
wait_for_log_token "/workspace/logs/install_setup.log" "=== INSTALL DONE ===" 1800 || die "Install job did not complete in time"
download_from_pod "/workspace/logs/install_setup.log" "${LOCAL_ARTIFACTS_DIR}/install_setup.log" 0 1 || true

log "Preparing data..."
DATA_SETUP_SCRIPT_CONTENT=$(cat <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/data/datasets/fineweb10B_sp1024 /workspace/data/tokenizers /workspace/logs
python3 - <<'PYEOF'
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
import os, pathlib, shutil, sys, time

repo = '${HF_REPO}'
data_dir = '/workspace/data/datasets/fineweb10B_sp1024'
tok_dir = '/workspace/data/tokenizers'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(tok_dir, exist_ok=True)

def fetch(subfolder, fname, dest):
    if os.path.exists(dest):
        return
    p = hf_hub_download(repo_id=repo, filename=fname, subfolder=subfolder, repo_type='dataset')
    src = pathlib.Path(p).resolve()
    try:
        os.link(src, dest)
    except Exception:
        shutil.copy2(src, dest)

fetch('datasets/tokenizers', 'fineweb_1024_bpe.model', os.path.join(tok_dir, 'fineweb_1024_bpe.model'))
fetch('datasets/datasets/fineweb10B_sp1024', 'fineweb_val_000000.bin', os.path.join(data_dir, 'fineweb_val_000000.bin'))

shards = [f'fineweb_train_{i:06d}.bin' for i in range(${DATA_SHARDS})]

def fetch_shard(fname):
    dest = os.path.join(data_dir, fname)
    if os.path.exists(dest):
        return f'{fname}: cached'
    for attempt in range(3):
        try:
            fetch('datasets/datasets/fineweb10B_sp1024', fname, dest)
            return f'{fname}: done'
        except Exception as e:
            if attempt == 2:
                return f'{fname}: FAILED ({e})'
            time.sleep(2 ** attempt)

failed = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(fetch_shard, shard): shard for shard in shards}
    for idx, fut in enumerate(as_completed(futs), start=1):
        result = fut.result()
        if 'FAILED' in result:
            failed += 1
            print(result, flush=True)
        if idx % 4 == 0 or idx == len(shards):
            print(f'{idx}/{len(shards)} train shards ready ({failed} failed)', flush=True)

actual = len([f for f in os.listdir(data_dir) if f.startswith('fineweb_train_')])
print(f'train_shards={actual}')
if actual < min(${DATA_SHARDS}, 6):
    sys.exit(1)
PYEOF
echo "=== DATA DONE ==="
EOF
)
write_remote_file "/workspace/data_setup.sh" "$DATA_SETUP_SCRIPT_CONTENT" | tee -a "$ORCH_LOG" || die "Failed to write data setup script"
launch_remote_job "small_setup_data" "/workspace/data_setup.sh" "/workspace/logs/data_setup.log" || die "Failed to launch data setup job"
wait_for_log_token "/workspace/logs/data_setup.log" "=== DATA DONE ===" 3600 || die "Data preparation failed"
download_from_pod "/workspace/logs/data_setup.log" "${LOCAL_ARTIFACTS_DIR}/data_setup.log" 0 1 || true

log "Uploading code..."
ul_retry "${SCRIPT_DIR}/train_gpt.py" "${SCRIPT_DIR}/run_small_skc_2gpu.sh" || die "Upload failed"

if [[ -n "${LATENT_CHECKPOINT_LOCAL_PATH}" ]]; then
    [[ -f "${LATENT_CHECKPOINT_LOCAL_PATH}" ]] || die "LATENT_CHECKPOINT_LOCAL_PATH not found: ${LATENT_CHECKPOINT_LOCAL_PATH}"
    log "Uploading latent checkpoint..."
    ul_retry "${LATENT_CHECKPOINT_LOCAL_PATH}" || die "Checkpoint upload failed"
    export LATENT_CHECKPOINT_PATH="/workspace/$(basename "${LATENT_CHECKPOINT_LOCAL_PATH}")"
fi

REMOTE_ENV_PREFIX="$(build_remote_env_prefix)"
log "Remote env overrides: ${REMOTE_ENV_PREFIX:-<none>}"

log "Smoke test..."
r_retry "cd /workspace && timeout ${SMOKE_TIMEOUT_SECONDS}s env ${REMOTE_ENV_PREFIX}FAST_SMOKE=1 NPROC_PER_NODE=${GPU_COUNT} GRAD_ACCUM_STEPS=1 bash run_small_skc_2gpu.sh 2>&1 | tee /workspace/logs/smoke.log" | tee -a "$ORCH_LOG" || die "Smoke failed"
download_from_pod "/workspace/logs/smoke.log" "${LOCAL_ARTIFACTS_DIR}/smoke.log" 0 1 || true

log "Launching 10-minute run..."
r_retry "tmux kill-session -t small_skc 2>/dev/null || true; cd /workspace && tmux new-session -d -s small_skc \
    \"env ${REMOTE_ENV_PREFIX}SEED=${SEED:-42} NPROC_PER_NODE=${GPU_COUNT} GRAD_ACCUM_STEPS=1 bash run_small_skc_2gpu.sh 2>&1 | tee /workspace/logs/full_run.log\""

wait_for_log_token "/workspace/logs/full_run.log" "=== DONE ===" 1800 || die "Full run did not complete in time"
download_from_pod "/workspace/logs/full_run.log" "${LOCAL_ARTIFACTS_DIR}/full_run.log" 0 1 || true

RUN_ID=$(grep "RUN_ID" "${LOCAL_ARTIFACTS_DIR}/full_run.log" 2>/dev/null | head -1 | grep -oE "small_skc_2gpu_s[0-9]+_[0-9_]+" | head -1)
if [[ -z "${RUN_ID:-}" ]]; then
    die "Could not parse RUN_ID from full run log"
fi
log "RUN_ID=${RUN_ID}"

download_run_artifacts "$RUN_ID"

if [[ -f "${DOWNLOAD_DIR}/${RUN_ID}_submission.json" ]]; then
    log "Submission summary:"
    jq '. | {model_size, val_loss, val_bpb}' "${DOWNLOAD_DIR}/${RUN_ID}_submission.json" | tee -a "$ORCH_LOG" || true
fi

terminate_pod "$POD_ID"

log "Complete. Artifacts:"
log "  log        : ${LOCAL_ARTIFACTS_DIR}/full_run.log"
log "  download dir: ${DOWNLOAD_DIR}"
