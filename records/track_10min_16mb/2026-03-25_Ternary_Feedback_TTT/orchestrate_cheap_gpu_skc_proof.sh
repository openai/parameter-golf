#!/usr/bin/env bash
# ============================================================================
# RunPod single-GPU SKC proof orchestrator
# Uses a cheap 1x GPU to run the current CUDA SKC stack on real FineWeb shards.
# ============================================================================
set -uo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"

LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="golf-cheap-proof-$(date +%Y%m%d-%H%M%S)"
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
GPU_TYPES=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA L40S")
GPU_COUNT=1
VOLUME_GB=30
DISK_GB=30
DATA_SHARDS="${DATA_SHARDS:-20}"
HF_REPO="willdepueoai/parameter-golf"
SCP_RETRIES="${SCP_RETRIES:-3}"
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
POD_TERMINATED=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ARTIFACTS_DIR="${LOCAL_ARTIFACTS_DIR:-${SCRIPT_DIR}/cheap_gpu_proof_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_ARTIFACTS_DIR"
ORCH_LOG="${LOCAL_ARTIFACTS_DIR}/orchestrator.log"

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$ORCH_LOG"; }
die() {
    log "FATAL: $*"
    [[ -n "${POD_ID:-}" ]] && terminate_pod "$POD_ID" || true
    exit 1
}
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

gql() {
    curl -s --request POST \
        --header "content-type: application/json" \
        --url "$RUNPOD_API" \
        --data "$(jq -n --arg q "$1" '{"query": $q}')"
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
    minVcpuCount: 2
    minMemoryInGb: 15
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
    log "  success: pod=${POD_ID} cost=\$$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.costPerHr // "unknown"')/hr"
    return 0
}

create_pod() {
    local gpu
    for gpu in "${GPU_TYPES[@]}"; do
        try_create_pod "$gpu" "COMMUNITY" && return 0
        try_create_pod "$gpu" "SECURE" && return 0
    done
    die "Could not create a cheap GPU pod"
}

terminate_pod() {
    local pid="${1:-${POD_ID:-}}"
    [[ -z "${pid:-}" || "$pid" == "null" ]] && return 0
    log "Terminating pod ${pid}..."
    gql "mutation { podTerminate(input: { podId: \"${pid}\" }) }" | jq -r '.data // .errors' | tee -a "$ORCH_LOG" || true
    POD_TERMINATED=1
}

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

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
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

log "========================================================"
log "Cheap-GPU SKC Proof Run"
log "Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "Data shards: ${DATA_SHARDS}"
log "========================================================"

require_cmd curl
require_cmd jq
require_cmd ssh
require_cmd scp
[[ -f "$LOCAL_SSH_KEY" ]] || die "SSH private key not found: $LOCAL_SSH_KEY"
[[ -f "$LOCAL_SSH_PUB" ]] || die "SSH public key not found: $LOCAL_SSH_PUB"
[[ -f "${SCRIPT_DIR}/train_gpt.py" ]] || die "train_gpt.py not found in $SCRIPT_DIR"
[[ -f "${SCRIPT_DIR}/run_cheap_gpu_skc_proof.sh" ]] || die "run_cheap_gpu_skc_proof.sh not found in $SCRIPT_DIR"

create_pod
echo "$POD_ID" > "${LOCAL_ARTIFACTS_DIR}/pod_id.txt"
wait_for_pod

log "Verifying GPU and runtime..."
r "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" | tee -a "$ORCH_LOG" || die "nvidia-smi failed"
r "python3 -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"gpus\", torch.cuda.device_count())'" | tee -a "$ORCH_LOG" || die "PyTorch unavailable"

log "Installing dependencies..."
r "pip install -q sentencepiece zstandard huggingface-hub && echo deps_ok" | tee -a "$ORCH_LOG" || die "pip install failed"

log "Downloading dataset..."
r "mkdir -p /workspace/data/datasets/fineweb10B_sp1024 /workspace/data/tokenizers /workspace/logs"
r "python3 - <<'PYEOF'
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
import os, pathlib, shutil

repo = '${HF_REPO}'
dest_dir = '/workspace/data/datasets/fineweb10B_sp1024'
tok_dir = '/workspace/data/tokenizers'
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(tok_dir, exist_ok=True)

def copy_from_hf(filename, subfolder, dest):
    if os.path.exists(dest):
        return f'{filename}: exists'
    p = hf_hub_download(repo_id=repo, filename=filename, subfolder=subfolder, repo_type='dataset')
    src = pathlib.Path(p).resolve()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return f'{filename}: ok'

print(copy_from_hf('fineweb_val_000000.bin', 'datasets/datasets/fineweb10B_sp1024', f'{dest_dir}/fineweb_val_000000.bin'))
print(copy_from_hf('fineweb_1024_bpe.model', 'datasets/tokenizers', f'{tok_dir}/fineweb_1024_bpe.model'))

shards = [f'fineweb_train_{i:06d}.bin' for i in range(${DATA_SHARDS})]
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(copy_from_hf, s, 'datasets/datasets/fineweb10B_sp1024', f'{dest_dir}/{s}') for s in shards]
    for i, fut in enumerate(as_completed(futs), 1):
        fut.result()
        if i % 5 == 0 or i == len(shards):
            print(f'shards: {i}/{len(shards)}', flush=True)
PYEOF" | tee -a "$ORCH_LOG" || die "dataset download failed"

log "Uploading code..."
ul "${SCRIPT_DIR}/train_gpt.py" "${SCRIPT_DIR}/run_cheap_gpu_skc_proof.sh" || die "code upload failed"
r "chmod +x /workspace/run_cheap_gpu_skc_proof.sh"

log "Smoke testing..."
r "cd /workspace && FAST_SMOKE=1 timeout 180 bash run_cheap_gpu_skc_proof.sh > /workspace/logs/cheap_gpu_smoke.log 2>&1" || true
download_from_pod "/workspace/logs/cheap_gpu_smoke.log" "${LOCAL_ARTIFACTS_DIR}/smoke.log" 0 1 || true
if ! rg -q "step:[0-9]" "${LOCAL_ARTIFACTS_DIR}/smoke.log" 2>/dev/null; then
    die "smoke test did not produce training steps"
fi

log "Starting full proof run..."
r "cd /workspace && timeout 2700 bash run_cheap_gpu_skc_proof.sh 2>&1 | tee /workspace/logs/cheap_gpu_proof_console.log" | tee "${LOCAL_ARTIFACTS_DIR}/proof_console.log" || true

log "Downloading logs and artifacts..."
download_from_pod "/workspace/logs" "${LOCAL_ARTIFACTS_DIR}/downloads" 1 1 || true
download_from_pod "/workspace/final_model.ternary.ptz" "${LOCAL_ARTIFACTS_DIR}/final_model.ternary.ptz" 0 1 || true
download_from_pod "/workspace/submission.json" "${LOCAL_ARTIFACTS_DIR}/submission.json" 0 1 || true

log "Summary:"
rg -n "step:[0-9]+/500000 val_loss:|final_ternary_roundtrip|final_sliding|ngram_cache|artifact:" "${LOCAL_ARTIFACTS_DIR}/proof_console.log" | tee -a "$ORCH_LOG" || true

terminate_pod "$POD_ID"
log "Done. Artifacts at ${LOCAL_ARTIFACTS_DIR}"
