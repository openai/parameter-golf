#!/usr/bin/env bash
# ============================================================================
# RunPod single-GPU compute-optimal size sweep
# Runs a depth sweep around the Chinchilla-like token budget on one cheap GPU.
# ============================================================================
set -uo pipefail

RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"

LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="golf-size-sweep-$(date +%Y%m%d-%H%M%S)"
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
GPU_TYPES=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA L40S")
GPU_COUNT=1
VOLUME_GB=30
DISK_GB=30
DATA_SHARDS="${DATA_SHARDS:-20}"
RUN_SECONDS="${RUN_SECONDS:-420}"
FULL_VALIDATE="${FULL_VALIDATE:-0}"
HF_REPO="willdepueoai/parameter-golf"
SCP_RETRIES="${SCP_RETRIES:-3}"
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
POD_TERMINATED=0

SIZE_CONFIGS=(
    "skc_6L_512D_128C"
    "skc_8L_512D_128C"
    "skc_10L_512D_128C"
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
LOCAL_ARTIFACTS_DIR="${LOCAL_ARTIFACTS_DIR:-${SCRIPT_DIR}/cheap_gpu_size_sweep_$(date +%Y%m%d_%H%M%S)}"
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

download_run_artifacts() {
    local config="$1"
    local run_dir="${LOCAL_ARTIFACTS_DIR}/${config}"
    mkdir -p "$run_dir"
    download_from_pod "/workspace/logs/ablation_${config}.log" "${run_dir}/ablation_${config}.log" 0 1 || true
    download_from_pod "/workspace/logs/ablation_${config}_submission.json" "${run_dir}/submission.json" 0 1 || true
    download_from_pod "/workspace/logs/ablation_${config}_model.ternary.ptz" "${run_dir}/model.ternary.ptz" 0 1 || true
}

summarize_results() {
    python3 - <<'PYEOF' "$LOCAL_ARTIFACTS_DIR" "${SIZE_CONFIGS[@]}"
import json, pathlib, re, sys

root = pathlib.Path(sys.argv[1])
configs = sys.argv[2:]
rows = []
for cfg in configs:
    run_dir = root / cfg
    submission = run_dir / "submission.json"
    log_path = run_dir / f"ablation_{cfg}.log"
    val_bpb = None
    params = None
    if submission.exists():
        try:
            val_bpb = json.loads(submission.read_text()).get("val_bpb")
        except Exception:
            val_bpb = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            m = re.search(r"params:(\d+)", line)
            if m:
                params = int(m.group(1))
                break
    rows.append((cfg, val_bpb, params))

rows.sort(key=lambda row: float("inf") if row[1] is None else row[1])
print("CONFIG\tFINAL_BPB\tPARAMS")
for cfg, val_bpb, params in rows:
    bpb_s = "NA" if val_bpb is None else f"{val_bpb:.4f}"
    param_s = "NA" if params is None else str(params)
    print(f"{cfg}\t{bpb_s}\t{param_s}")
PYEOF
}

log "========================================================"
log "Cheap-GPU Compute-Optimal Size Sweep"
log "Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "Data shards: ${DATA_SHARDS}"
log "Run seconds/config: ${RUN_SECONDS}"
log "Eval mode: $( [[ "$FULL_VALIDATE" == "1" ]] && echo "full_validate" || echo "proxy_roundtrip" )"
log "Configs: ${SIZE_CONFIGS[*]}"
log "========================================================"

require_cmd curl
require_cmd jq
require_cmd ssh
require_cmd scp
require_cmd python3
[[ -f "$LOCAL_SSH_KEY" ]] || die "SSH private key not found: $LOCAL_SSH_KEY"
[[ -f "$LOCAL_SSH_PUB" ]] || die "SSH public key not found: $LOCAL_SSH_PUB"
[[ -f "${TRAINER_PATH}" ]] || die "train_gpt.py not found in $SCRIPT_DIR"
[[ -f "${SCRIPT_DIR}/run_single_ablation.sh" ]] || die "run_single_ablation.sh not found in $SCRIPT_DIR"

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
ul "${TRAINER_PATH}" "${SCRIPT_DIR}/run_single_ablation.sh" || die "code upload failed"
r "chmod +x /workspace/run_single_ablation.sh"

log "Smoke testing size sweep path..."
r "cd /workspace && FAST_SMOKE=1 MODE=screen timeout 180 bash run_single_ablation.sh skc_6L_512D_128C > /workspace/logs/size_sweep_smoke.log 2>&1" || true
download_from_pod "/workspace/logs/size_sweep_smoke.log" "${LOCAL_ARTIFACTS_DIR}/smoke.log" 0 1 || true
if ! rg -q "step:[0-9]" "${LOCAL_ARTIFACTS_DIR}/smoke.log" 2>/dev/null; then
    die "smoke test did not produce training steps"
fi

for config in "${SIZE_CONFIGS[@]}"; do
    log "Starting size config: ${config}"
    mkdir -p "${LOCAL_ARTIFACTS_DIR}/${config}"
    if [[ "$FULL_VALIDATE" == "1" ]]; then
        RUN_CMD="cd /workspace && MAX_WALLCLOCK_SECONDS=${RUN_SECONDS} MODE=validate bash run_single_ablation.sh ${config} 2>&1 | tee /workspace/logs/console_${config}.log"
    else
        RUN_CMD="cd /workspace && MAX_WALLCLOCK_SECONDS=${RUN_SECONDS} MODE=validate GPTQ_LITE_ENABLED=0 HESSIAN_TERNARY_GPTQ=0 SELECTIVE_PRUNING=0 SLIDING_EVAL=0 NGRAM_CACHE_ENABLED=0 TEMP_SCALING=0 bash run_single_ablation.sh ${config} 2>&1 | tee /workspace/logs/console_${config}.log"
    fi
    r "$RUN_CMD" \
        | tee "${LOCAL_ARTIFACTS_DIR}/${config}/console.log" || true
    download_run_artifacts "$config"
done

log "Downloading full logs bundle..."
download_from_pod "/workspace/logs" "${LOCAL_ARTIFACTS_DIR}/downloads" 1 1 || true

log "Ranking by exported BPB..."
summarize_results | tee "${LOCAL_ARTIFACTS_DIR}/summary.tsv" | tee -a "$ORCH_LOG"

terminate_pod "$POD_ID"
log "Done. Artifacts at ${LOCAL_ARTIFACTS_DIR}"
