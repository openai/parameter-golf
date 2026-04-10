#!/usr/bin/env bash
# ============================================================================
# Parameter Golf — Aligned Architecture Shootout (128-Aligned)
# ============================================================================
set -uo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"
LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
GPU_COUNT=1
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="golf-aligned-shootout-$(date +%Y%m%d-%H%M%S)"
VOLUME_GB=30
DISK_GB=30
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
DATA_SHARDS=10
HF_REPO="willdepueoai/parameter-golf"

GPU_TYPES=("NVIDIA L40S" "NVIDIA A40" "NVIDIA RTX A6000")

# Aligned Architecture shootout (Validation Mode)
ABLATION_CONFIGS=(
    skc_10L_640D_128C
    skc_12L_512D_128C
    skc_10L_640D_256C
    skc_12L_640D_128C
)

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
LOCAL_ARTIFACTS_DIR="${SCRIPT_DIR}/aligned_shootout_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOCAL_ARTIFACTS_DIR"
ORCH_LOG="${LOCAL_ARTIFACTS_DIR}/orchestrator.log"

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$ORCH_LOG"; }
die() { log "FATAL: $*"; [[ -n "${POD_ID:-}" ]] && terminate_pod "$POD_ID" || true; exit 1; }

gql() { curl -s --request POST --header 'content-type: application/json' --url "$RUNPOD_API" --data "$(jq -n --arg q "$1" '{"query": $q}')"; }

try_create_pod() {
    local gpu_type="$1" cloud="$2" pub_key=$(cat "$LOCAL_SSH_PUB")
    log "  Trying: ${GPU_COUNT}×${gpu_type} on ${cloud}..."
    local mutation="mutation { podFindAndDeployOnDemand(input: { gpuTypeId: \"${gpu_type}\", cloudType: ${cloud}, gpuCount: ${GPU_COUNT}, volumeInGb: ${VOLUME_GB}, containerDiskInGb: ${DISK_GB}, imageName: \"${POD_IMAGE}\", name: \"${POD_NAME}\", ports: \"22/tcp\", volumeMountPath: \"/workspace\", env: [{ key: \"PUBLIC_KEY\", value: \"${pub_key}\" }] }) { id costPerHr } }"
    local result=$(gql "$mutation")
    POD_ID=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
    [[ -z "$POD_ID" ]] && return 1
    log "  SUCCESS: pod=${POD_ID}"
    return 0
}

terminate_pod() { log "Terminating pod ${1}..."; gql "mutation { podTerminate(input: { podId: \"${1}\" }) }"; }

wait_for_pod() {
    log "Waiting for pod readiness..."
    for i in {1..30}; do
        sleep 10
        local res=$(gql "query { pod(input: { podId: \"${POD_ID}\" }) { runtime { ports { ip isIpPublic privatePort publicPort } } } }")
        POD_HOST=$(echo "$res" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22 and .isIpPublic == true) | .ip')
        POD_PORT=$(echo "$res" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22 and .isIpPublic == true) | .publicPort')
        if [[ -n "$POD_HOST" && "$POD_HOST" != "null" ]]; then
            if ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 -i "$LOCAL_SSH_KEY" -p "$POD_PORT" "root@${POD_HOST}" "echo OK" &>/dev/null; then
                log "Pod ready at root@${POD_HOST}:${POD_PORT}"
                return 0
            fi
        fi
    done
    die "Pod timeout"
}

# ── Main ─────────────────────────────────────────────────────────────────────
for gpu in "${GPU_TYPES[@]}"; do try_create_pod "$gpu" "COMMUNITY" && break; done
[[ -z "${POD_ID:-}" ]] && die "No GPU available"
wait_for_pod

SSH_BASE=("ssh" "-o" "StrictHostKeyChecking=no" "-i" "$LOCAL_SSH_KEY" "-p" "$POD_PORT" "root@${POD_HOST}")
SCP_BASE=("scp" "-o" "StrictHostKeyChecking=no" "-i" "$LOCAL_SSH_KEY" "-P" "$POD_PORT")

log "Downloading data..."
"${SSH_BASE[@]}" "mkdir -p /workspace/data/datasets/fineweb10B_sp1024 /workspace/data/tokenizers"
"${SSH_BASE[@]}" "pip install -q huggingface-hub sentencepiece zstandard"
log "Running data download script..."
"${SSH_BASE[@]}" "python3 -c \"
from huggingface_hub import hf_hub_download
import os, shutil
repo = '${HF_REPO}'
def dl(f, d):
    if os.path.exists(d): return
    p = hf_hub_download(repo_id=repo, filename=f, subfolder='datasets/'+os.path.dirname(d.replace('/workspace/data/','')), repo_type='dataset')
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(p, d)
dl('fineweb_val_000000.bin', '/workspace/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin')
dl('fineweb_1024_bpe.model', '/workspace/data/tokenizers/fineweb_1024_bpe.model')
for i in range(${DATA_SHARDS}):
    dl(f'fineweb_train_{i:06d}.bin', f'/workspace/data/datasets/fineweb10B_sp1024/fineweb_train_{i:06d}.bin')
\""

log "Uploading code..."
"${SCP_BASE[@]}" "${TRAINER_PATH}" "${SCRIPT_DIR}/run_single_ablation.sh" root@${POD_HOST}:/workspace/
"${SSH_BASE[@]}" "chmod +x /workspace/run_single_ablation.sh && mkdir -p /workspace/logs"

RESULTS=()
for config in "${ABLATION_CONFIGS[@]}"; do
    log "Starting aligned shootout: ${config} (MODE=validate)"
    "${SSH_BASE[@]}" "MODE=validate bash /workspace/run_single_ablation.sh ${config}" | tee "${LOCAL_ARTIFACTS_DIR}/ablation_${config}.log"
    
    # Extract metrics
    s_bpb=$(grep "val_sliding_bpb" "${LOCAL_ARTIFACTS_DIR}/ablation_${config}.log" | tail -1 | awk -F': ' '{print $2}')
    n_bpb=$(grep "val_ngram_bpb" "${LOCAL_ARTIFACTS_DIR}/ablation_${config}.log" | tail -1 | awk -F': ' '{print $2}')
    params=$(grep "params:[0-9]\+" "${LOCAL_ARTIFACTS_DIR}/ablation_${config}.log" | tail -1 | grep -oE "params:[0-9]+" | cut -d: -f2 || echo "N/A")
    RESULTS+=("${config}|${s_bpb}|${n_bpb}|${params}")
done

log "Final Shootout Results:"
printf "%-20s | %-12s | %-12s | %-12s\n" "CONFIG" "SLIDING" "NGRAM" "PARAMS"
for r in "${RESULTS[@]}"; do
    IFS='|' read -r c s n p <<< "$r"
    printf "%-20s | %-12s | %-12s | %-12s\n" "$c" "$s" "$n" "$p"
done

terminate_pod "$POD_ID"
log " shootout complete."
