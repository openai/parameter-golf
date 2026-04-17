#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/akhileshgogikar/parameter-golf"
[[ -f "${PROJECT_ROOT}/.runpod_secret.sh" ]] && source "${PROJECT_ROOT}/.runpod_secret.sh"

RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"
LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="paramgolf-skc-matrix-$(date +%Y%m%d-%H%M%S)"
RUNPOD_API="https://api.runpod.io/graphql"
GPU_COUNT="${GPU_COUNT:-2}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-22}"
VOLUME_GB="${VOLUME_GB:-20}"
DISK_GB="${DISK_GB:-20}"
DATA_SHARDS="${DATA_SHARDS:-24}"
HF_REPO="kevclark/parameter-golf"
BALANCE_FLOOR="${BALANCE_FLOOR:-10}"
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
POD_TERMINATED=0
POD_ID=""
POD_HOST=""
POD_PORT=""

LOCAL_ARTIFACTS_DIR="${LOCAL_ARTIFACTS_DIR:-${PROJECT_ROOT}/logs/skc_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_ARTIFACTS_DIR"
ORCH_LOG="${LOCAL_ARTIFACTS_DIR}/orchestrator.log"

log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$ORCH_LOG"; }
die() { log "FATAL: $*"; [[ -n "${POD_ID:-}" && "$POD_TERMINATED" -eq 0 ]] && terminate_pod; exit 1; }

SSH_BASE=(-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=5 -o BatchMode=yes -i "$LOCAL_SSH_KEY")
r()   { ssh "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "$@"; }
ul()  { scp "${SSH_BASE[@]}" -P "$POD_PORT" "$@" "root@${POD_HOST}:/workspace/"; }

gql() {
    curl -s --request POST \
        --header "content-type: application/json" \
        --header "Authorization: ${RUNPOD_API_KEY}" \
        --url "$RUNPOD_API" \
        --data "$(jq -n --arg q "$1" '{"query": $q}')"
}

get_balance() { gql 'query { myself { clientBalance } }' | jq -r '.data.myself.clientBalance // empty'; }

terminate_pod() {
    [[ -z "${POD_ID:-}" || "$POD_ID" == "null" ]] && return 0
    [[ "$KEEP_POD_ON_EXIT" == "1" ]] && { log "KEEP_POD_ON_EXIT=1, leaving pod ${POD_ID} running."; return 0; }
    log "Terminating pod ${POD_ID}..."
    gql "mutation { podTerminate(input: { podId: \"${POD_ID}\" }) }" >/dev/null || true
    POD_TERMINATED=1
}
trap terminate_pod EXIT INT TERM

refresh_pod_endpoint() {
    local q host port
    q="query { pod(input: { podId: \"${POD_ID}\" }) { runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }"
    host=$(gql "$q" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .ip' | head -1)
    port=$(gql "$q" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .publicPort' | head -1)
    [[ -z "${host:-}" || "$host" == "null" || -z "${port:-}" || "$port" == "null" ]] && return 1
    POD_HOST="$host"; POD_PORT="$port"
}

r_retry() {
    for (( attempt = 1; attempt <= 10; attempt++ )); do
        refresh_pod_endpoint >/dev/null 2>&1 || true
        r "$@" && return 0
        log "  SSH attempt ${attempt}/10 failed, retrying in 15s..."
        sleep 15
    done
    return 1
}

ul_retry() {
    for (( attempt = 1; attempt <= 3; attempt++ )); do
        refresh_pod_endpoint >/dev/null 2>&1 || true
        ul "$@" && return 0
        sleep 5
    done
    return 1
}

download_from_pod() {
    local remote_path=$1 local_path=$2 optional=${3:-0}
    for (( attempt = 1; attempt <= 3; attempt++ )); do
        refresh_pod_endpoint >/dev/null 2>&1 || true
        scp "${SSH_BASE[@]}" -P "$POD_PORT" "root@${POD_HOST}:${remote_path}" "$local_path" >/dev/null 2>&1 && return 0
        sleep 3
    done
    [[ "$optional" == "1" ]] && return 1
    log "WARNING: failed to download ${remote_path}"
    return 1
}

write_remote_file() {
    local remote_path="$1" content="$2"
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

wait_for_log_token() {
    local remote_log=$1 pattern=$2 timeout=${3:-1800}
    local waited=0 tmp="${LOCAL_ARTIFACTS_DIR}/poll.log"
    while [[ $waited -lt $timeout ]]; do
        sleep 20
        waited=$(( waited + 20 ))
        download_from_pod "$remote_log" "$tmp" 1 || true
        if [[ -f "$tmp" ]] && grep -q "$pattern" "$tmp"; then
            return 0
        fi
        if [[ -f "$tmp" ]]; then
            local line
            line=$(grep -E "=== START|=== END|step:|val_bpb:|final_sliding|train_batch_tokens|run_id=" "$tmp" | tail -1 || true)
            [[ -n "${line:-}" ]] && log "progress: ${line}"
        fi
    done
    return 1
}

pick_gpu_candidates() {
    gql "query {
      gpuTypes {
        id displayName memoryInGb
        lowestPrice(input: { gpuCount: ${GPU_COUNT} }) {
          uninterruptablePrice
          stockStatus
          minimumBidPrice
        }
      }
    }" | jq -r \
    --argjson minmem "$MIN_GPU_MEMORY_GB" '
      .data.gpuTypes[]
        | select(.memoryInGb >= $minmem)
        | select(.lowestPrice != null)
        | select((.displayName | test("V100|P100|T4|K80|M60|A2|A16|MI|Radeon|Instinct|Blackwell|PRO 6000|5090|5080|5070|5060"; "i")) | not)
        | [.lowestPrice.uninterruptablePrice, .id, .lowestPrice.stockStatus, (.memoryInGb | tostring), .displayName]
        | @tsv
    ' | sort -n
}

create_pod() {
    local candidates pub_key
    candidates=$(pick_gpu_candidates)
    [[ -n "$candidates" ]] || die "No viable ${GPU_COUNT}-GPU candidates found"
    log "GPU candidates:"
    printf "%s\n" "$candidates" | tee -a "$ORCH_LOG"

    pub_key=$(cat "$LOCAL_SSH_PUB") || die "Cannot read SSH public key"
    while IFS=$'\t' read -r price gpu_id stock mem display_name; do
        [[ -z "${gpu_id:-}" ]] && continue
        log "Trying ${GPU_COUNT}x ${display_name} (${gpu_id}) \$${price}/hr stock=${stock}..."
        local mutation result
        mutation="mutation {
          podFindAndDeployOnDemand(input: {
            gpuTypeId: \"${gpu_id}\"
            cloudType: COMMUNITY
            gpuCount: ${GPU_COUNT}
            volumeInGb: ${VOLUME_GB}
            containerDiskInGb: ${DISK_GB}
            minVcpuCount: 1
            minMemoryInGb: 8
            imageName: \"${POD_IMAGE}\"
            name: \"${POD_NAME}\"
            ports: \"22/tcp\"
            volumeMountPath: \"/workspace\"
            env: [{ key: \"PUBLIC_KEY\", value: \"${pub_key}\" }]
          }) { id costPerHr }
        }"
        result=$(gql "$mutation")
        if echo "$result" | jq -e '.errors' >/dev/null 2>&1; then
            log "  failed: $(echo "$result" | jq -r '.errors[].message' | head -1)"
            continue
        fi
        POD_ID=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.id // empty')
        [[ -z "$POD_ID" ]] && continue
        local cost
        cost=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.costPerHr // "unknown"')
        log "  SUCCESS: pod=${POD_ID} cost=\$${cost}/hr"
        return 0
    done <<< "$candidates"
    die "Could not create pod from any candidate"
}

wait_for_pod() {
    log "Waiting for pod runtime + SSH endpoint..."
    for ((i=1; i<=80; i++)); do
        if refresh_pod_endpoint >/dev/null 2>&1; then
            if r "echo SSH_OK" >/dev/null 2>&1; then
                log "Pod ready: ${POD_HOST}:${POD_PORT}"
                return 0
            fi
        fi
        sleep 15
    done
    die "Pod did not become reachable"
}

log "========================================================"
log "SKC Matrix RunPod Orchestrator (E1-E5)"
log "Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "GPUs: ${GPU_COUNT}  Min VRAM: ${MIN_GPU_MEMORY_GB}GB"
log "========================================================"

for cmd in curl jq ssh scp python3; do command -v "$cmd" >/dev/null 2>&1 || die "Required: $cmd"; done
[[ -f "$LOCAL_SSH_KEY" ]] || die "SSH key not found: $LOCAL_SSH_KEY"
[[ -f "$LOCAL_SSH_PUB" ]] || die "SSH pubkey not found: $LOCAL_SSH_PUB"
[[ -f "${PROJECT_ROOT}/run_god_tier_skc.sh" ]] || die "run_god_tier_skc.sh not found"
[[ -f "${PROJECT_ROOT}/train_gpt_verbose.py" ]] || die "train_gpt_verbose.py not found"
[[ -f "${PROJECT_ROOT}/build_submission.py" ]] || die "build_submission.py not found"
[[ -f "${PROJECT_ROOT}/triton_kernels.py" ]] || die "triton_kernels.py not found"

balance=$(get_balance)
log "Balance: \$${balance}"
awk -v b="$balance" -v f="$BALANCE_FLOOR" 'BEGIN { exit !(b <= f) }' && die "Balance too low: \$${balance} <= \$${BALANCE_FLOOR}"

create_pod
echo "$POD_ID" > "${LOCAL_ARTIFACTS_DIR}/pod_id.txt"
wait_for_pod

log "Verifying GPU..."
r_retry "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" | tee -a "$ORCH_LOG"

log "Installing deps + tmux..."
r_retry "which tmux || (export DEBIAN_FRONTEND=noninteractive; apt-get update -qq || true; apt-get install -y -q -o DPkg::Lock::Timeout=300 tmux); pip install -q sentencepiece zstandard huggingface-hub && echo DEPS_OK" | tee -a "$ORCH_LOG"

log "Downloading data..."
DATA_SCRIPT=$(cat <<DATAEOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/data/datasets/fineweb10B_sp8192 /workspace/data/tokenizers /workspace/logs /workspace/cache/torch /workspace/cache/triton
python3 - <<'PYEOF'
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
import os, pathlib, shutil, time

repo = '${HF_REPO}'
data_dir = '/workspace/data/datasets/fineweb10B_sp8192'
tok_dir = '/workspace/data/tokenizers'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(tok_dir, exist_ok=True)

for fname, url in [
    ('fineweb_8192_bpe.model', 'https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets/tokenizers/fineweb_8192_bpe.model?download=true'),
    ('fineweb_8192_bpe.vocab', 'https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets/tokenizers/fineweb_8192_bpe.vocab?download=true'),
    ('tokenizer_specs_8192.json', 'https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets/tokenizers/tokenizer_specs_8192.json?download=true'),
]:
    dest = os.path.join(tok_dir, fname)
    if not os.path.exists(dest):
        os.system(f"curl -L -s -o {dest} '{url}'")
        print(f"tokenizer: {fname}", flush=True)

def fetch(fname):
    dest = os.path.join(data_dir, fname)
    if os.path.exists(dest): return f'{fname}: cached'
    for attempt in range(3):
        try:
            p = hf_hub_download(repo_id=repo, filename=fname, subfolder='datasets/datasets/fineweb10B_sp8192', repo_type='dataset')
            src = pathlib.Path(p).resolve()
            try: os.link(src, dest)
            except: shutil.copy2(src, dest)
            return f'{fname}: done'
        except Exception as e:
            if attempt == 2: raise
            time.sleep(2)

fetch('fineweb_val_000000.bin')
shards = ['fineweb_val_000000.bin'] + [f'fineweb_train_{i:06d}.bin' for i in range(${DATA_SHARDS})]
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = {ex.submit(fetch, s): s for s in shards}
    for i, fut in enumerate(as_completed(futs), 1):
        r = fut.result()
        if i % 4 == 0 or i == len(shards): print(f'{i}/{len(shards)}: {r}', flush=True)
PYEOF
echo "=== DATA DONE ==="
DATAEOF
)
write_remote_file "/workspace/data_setup.sh" "$DATA_SCRIPT"
r_retry "mkdir -p /workspace/logs && tmux new-session -d -s data_dl 'bash /workspace/data_setup.sh > /workspace/logs/data_setup.log 2>&1'" || die "Failed to start data download"

log "Uploading code..."
ul_retry \
    "${PROJECT_ROOT}/run_god_tier_skc.sh" \
    "${PROJECT_ROOT}/build_submission.py" \
    "${PROJECT_ROOT}/train_gpt_verbose.py" \
    "${PROJECT_ROOT}/train_gpt.py" \
    "${PROJECT_ROOT}/triton_kernels.py" || die "Upload failed"
r_retry "chmod +x /workspace/*.sh /workspace/*.py" || true

log "Waiting for data..."
wait_for_log_token "/workspace/logs/data_setup.log" "=== DATA DONE ===" 3600 || die "Data download failed"
download_from_pod "/workspace/logs/data_setup.log" "${LOCAL_ARTIFACTS_DIR}/data_setup.log" 1 || true

log "Launching E1-E5 matrix..."
MATRIX_SCRIPT=$(cat <<'MATRIXEOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p /workspace/logs/cuda /workspace/logs/max_vram_10min

run_one() {
  local rid="$1"; shift
  echo "=== START ${rid} $(date -Is) ===" | tee -a /workspace/logs/cuda/${rid}.launcher.log
  env "$@" RUN_ID="$rid" bash /workspace/run_god_tier_skc.sh >> /workspace/logs/cuda/${rid}.txt 2>&1
  echo "=== END ${rid} $(date -Is) ===" | tee -a /workspace/logs/cuda/${rid}.launcher.log
}

common=(
  AUTO_TUNE=0
  DIAGNOSTICS_ENABLED=1
  COMPILE_MODE=none
  DDP_FIND_UNUSED_PARAMETERS=1
  TRAIN_LOG_EVERY=1
  TRAIN_LOG_EVERY_FRACTION=0
  VAL_LOSS_EVERY=50
  VAL_LOSS_EVERY_FRACTION=0
  EXPORT_ALIGNED_TRAIN=0
  EXPORT_PROXY_EVAL=0
  ITERATIONS=200000
  MAX_WALLCLOCK_SECONDS=570
  TRAIN_BATCH_TOKENS=10128
  TRITON_ENGRAM_ENABLED=0
  RUNTIME_PATH_POLICY=legacy
  ADAM_WD=0.0
)

run_one E1_control_legacy_wd0_sc1_r35_eng1 "${common[@]}" SCALES_LR_MULT=1.0 RECURRENCE_START_FRACTION=0.35 BIGRAM_HASH_ENABLED=1
run_one E2_scales_legacy_wd0_sc3_r35_eng1 "${common[@]}" SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.35 BIGRAM_HASH_ENABLED=1
run_one E3_recur_legacy_wd0_sc3_r20_eng1  "${common[@]}" SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.20 BIGRAM_HASH_ENABLED=1
run_one E4_engoff_legacy_wd0_sc3_r20_eng0 "${common[@]}" SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.20 BIGRAM_HASH_ENABLED=0
run_one E5_strict_wd0_sc3_r20_eng1         "${common[@]}" RUNTIME_PATH_POLICY=strict SCALES_LR_MULT=3.0 RECURRENCE_START_FRACTION=0.20 BIGRAM_HASH_ENABLED=1

echo "=== MATRIX DONE ==="
MATRIXEOF
)
write_remote_file "/workspace/launch_skc_matrix.sh" "$MATRIX_SCRIPT"
r_retry "tmux kill-session -t skc_matrix 2>/dev/null || true; tmux new-session -d -s skc_matrix 'bash /workspace/launch_skc_matrix.sh > /workspace/logs/skc_matrix_run.log 2>&1'"

log "Monitoring matrix (timeout 9000s)..."
wait_for_log_token "/workspace/logs/skc_matrix_run.log" "=== MATRIX DONE ===" 9000 || die "Matrix run did not complete"

log "Downloading results..."
download_from_pod "/workspace/logs/skc_matrix_run.log" "${LOCAL_ARTIFACTS_DIR}/skc_matrix_run.log" 1 || true
r_retry "ls /workspace/logs/cuda/ 2>/dev/null" | while read -r f; do
    [[ -z "$f" ]] && continue
    download_from_pod "/workspace/logs/cuda/$f" "${LOCAL_ARTIFACTS_DIR}/$f" 1 || true
done
r_retry "ls /workspace/logs/max_vram_10min/ 2>/dev/null" | while read -r f; do
    [[ -z "$f" ]] && continue
    download_from_pod "/workspace/logs/max_vram_10min/$f" "${LOCAL_ARTIFACTS_DIR}/$f" 1 || true
done

if [[ -f "${LOCAL_ARTIFACTS_DIR}/skc_matrix_run.log" ]]; then
    log ""
    log "========================================================"
    log "MATRIX RESULTS (tail)"
    log "========================================================"
    grep -E "=== START|=== END|val_bpb:|final_sliding|step: " "${LOCAL_ARTIFACTS_DIR}/skc_matrix_run.log" | tail -120 | tee -a "$ORCH_LOG" || true
    log "========================================================"
fi

terminate_pod
sleep 5
STATUS=$(gql "query { pod(input: { podId: \"${POD_ID}\" }) { desiredStatus } }" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
log "Pod final desiredStatus: ${STATUS}"

trap - EXIT INT TERM
log "Complete. Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "  Orchestrator log: ${ORCH_LOG}"
log "  Run log: ${LOCAL_ARTIFACTS_DIR}/skc_matrix_run.log"
