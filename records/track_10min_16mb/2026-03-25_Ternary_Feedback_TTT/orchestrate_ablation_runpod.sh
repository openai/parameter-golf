#!/usr/bin/env bash
# ============================================================================
# Parameter Golf — RunPod Ablation Orchestrator (Single Cheap GPU)
#
# Launches 20 ablation configs on a single CUDA GPU (A40/A100/etc),
# each with 5-minute training budget. Designed to run unattended.
#
# Budget: $25 max. At A40 pricing (~$0.39/hr) → ~64 hours max pod time.
# Estimated runtime: ~2.5 hours (~$1 at A40 pricing).
#
# Usage:
#   export RUNPOD_API_KEY=<your key>
#   bash orchestrate_ablation_runpod.sh
# ============================================================================
set -uo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: export RUNPOD_API_KEY=<your key>}"

# ── Configurable ─────────────────────────────────────────────────────────────
LOCAL_SSH_KEY="${LOCAL_SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL_SSH_PUB="${LOCAL_SSH_KEY}.pub"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
GPU_COUNT=1
POD_IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME="paramgolf-ablation-$(date +%Y%m%d-%H%M%S)"
VOLUME_GB=30
DISK_GB=30
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"
DATA_SHARDS=10
HF_REPO="willdepueoai/parameter-golf"
SCP_RETRIES=3
KEEP_POD_ON_EXIT="${KEEP_POD_ON_EXIT:-0}"
POD_TERMINATED=0

# GPU types to try, cheapest first (all support bf16/sm_80+)
GPU_TYPES=(
    "NVIDIA A40"
    "NVIDIA RTX A6000"
    "NVIDIA L40S"
    "NVIDIA A100 80GB PCIe"
    "NVIDIA A100-SXM4-80GB"
)

# Ablation configs in priority order (most important first)
ABLATION_CONFIGS=(
    baseline
    no_curriculum
    no_bigram
    no_xsa
    depth_8
    lr_high
    lr_low
    no_lawa
    no_swa
    no_smeargate
    no_partial_rope
    no_ln_damping
    warmdown_30
    warmdown_70
    no_engram_orders
    capsule_on
    vrl_on
    feedback_on
    depth_13
    dim_384
)

# ── Local paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ARTIFACTS_DIR="${LOCAL_ARTIFACTS_DIR:-${SCRIPT_DIR}/ablation_results_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_ARTIFACTS_DIR"
ORCH_LOG="${LOCAL_ARTIFACTS_DIR}/orchestrator.log"

# ── Budget (hard cap: 3 hours = well within $25 at any GPU price) ────────────
T_START=""
T_BUDGET=10800  # 3 hours max pod time

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

# ── RunPod API helper ─────────────────────────────────────────────────────────
gql() {
    curl -s --request POST \
        --header 'content-type: application/json' \
        --url "$RUNPOD_API" \
        --data "$(jq -n --arg q "$1" '{"query": $q}')"
}

# ── Pod lifecycle ─────────────────────────────────────────────────────────────
try_create_pod() {
    local gpu_type="$1"
    local cloud="$2"
    local pub_key
    pub_key=$(cat "$LOCAL_SSH_PUB") || die "Cannot read SSH public key: $LOCAL_SSH_PUB"

    log "  Trying: ${GPU_COUNT}×${gpu_type} on ${cloud}..."

    local mutation
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
    id name desiredStatus costPerHr
    runtime { ports { ip isIpPublic privatePort publicPort type } }
  }
}
EOF
)
    local result
    result=$(gql "$mutation")

    if echo "$result" | jq -e '.errors' > /dev/null 2>&1; then
        local err_msg
        err_msg=$(echo "$result" | jq -r '.errors[].message' 2>/dev/null | head -1)
        log "    FAILED: $err_msg"
        return 1
    fi

    POD_ID=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.id')
    local cost_hr
    cost_hr=$(echo "$result" | jq -r '.data.podFindAndDeployOnDemand.costPerHr // "unknown"')
    [[ -z "$POD_ID" || "$POD_ID" == "null" ]] && { log "    No pod ID in response"; return 1; }

    echo "$POD_ID" > "${LOCAL_ARTIFACTS_DIR}/pod_id.txt"
    echo "$gpu_type" > "${LOCAL_ARTIFACTS_DIR}/gpu_type.txt"
    log "  SUCCESS: pod=${POD_ID} gpu=${gpu_type} cost=\$${cost_hr}/hr"
    return 0
}

create_pod() {
    log "Creating pod (trying cheapest GPU first)..."
    for gpu_type in "${GPU_TYPES[@]}"; do
        try_create_pod "$gpu_type" "COMMUNITY" && return 0
        try_create_pod "$gpu_type" "SECURE" && return 0
    done
    die "Could not create pod with any GPU type"
}

wait_for_pod() {
    log "Waiting for pod SSH readiness..."
    local waited=0 MAX_WAIT=600
    while [[ $waited -lt $MAX_WAIT ]]; do
        sleep 10; waited=$(( waited + 10 ))
        refresh_pod_endpoint || { log "  waiting... ${waited}s"; continue; }
        if ssh "${SSH_BASE[@]}" -p "$POD_PORT" "root@${POD_HOST}" "echo SSH_OK" 2>/dev/null | grep -q "SSH_OK"; then
            log "Pod ready: root@${POD_HOST}:${POD_PORT}  T+$(t_elapsed)s"
            return 0
        fi
        log "  SSH not responding yet... ${waited}s"
    done
    die "Pod not SSH-ready after ${MAX_WAIT}s"
}

terminate_pod() {
    local pid="${1:-${POD_ID:-}}"
    [[ -z "${pid:-}" || "$pid" == "null" ]] && return 0
    log "Terminating pod ${pid}..."
    gql "mutation { podTerminate(input: { podId: \"${pid}\" }) }" | jq -r '.data // .errors' | tee -a "$ORCH_LOG" || true
    POD_TERMINATED=1
}

# ── SSH helpers ───────────────────────────────────────────────────────────────
SSH_BASE=(
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=15
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=10
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

download_from_pod() {
    local remote_path=$1 local_path=$2 recursive=${3:-0} optional=${4:-0}
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
    log "WARNING: failed to download ${remote_path} after ${SCP_RETRIES} attempts"
    return 1
}

# ── Ablation runner ───────────────────────────────────────────────────────────
run_ablation() {
    local config=$1
    local config_budget=480  # 8 min max per config (5 min train + overhead)
    local label="ablation_${config}"
    local remote_log="/workspace/logs/${label}.log"
    local local_log="${LOCAL_ARTIFACTS_DIR}/${label}.log"

    log "--- ${config}: starting ($(t_remaining)s left) ---"
    check_budget "$config" "$config_budget" || { log "SKIP $config — not enough time"; return 0; }

    # Run the ablation (blocking SSH — simpler than tmux for sequential runs)
    r "cd /workspace && timeout 420 bash run_single_ablation.sh ${config} 2>&1" > "$local_log" 2>&1 || true

    # Check for errors
    local has_nan has_err
    has_nan=$(grep -cE "val_bpb:nan|loss:nan" "$local_log" 2>/dev/null || echo 0)
    has_err=$(grep -cE "^Traceback|CUDA error|RuntimeError.*OOM|torch.cuda.OutOfMemoryError" "$local_log" 2>/dev/null || echo 0)

    # Extract BPB
    local final_bpb
    final_bpb=$(grep -E "val_bpb:" "$local_log" 2>/dev/null | tail -1 | grep -oE "val_bpb:[0-9.]+" | cut -d: -f2 || echo "N/A")

    if [[ "$has_nan" -gt 0 ]]; then
        log "${config}: NaN detected → BPB=NaN"
        RESULTS+=("${config}|NaN|ERROR")
    elif [[ "$has_err" -gt 0 ]]; then
        local err_type
        err_type=$(grep -oE "RuntimeError|OOM|OutOfMemory|AttributeError|KeyError" "$local_log" 2>/dev/null | head -1 || echo "unknown")
        log "${config}: ERROR (${err_type}) → skipped"
        RESULTS+=("${config}|ERROR|${err_type}")
    else
        log "${config}: DONE → val_bpb=${final_bpb}"
        RESULTS+=("${config}|${final_bpb}|OK")
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
log "========================================================"
log "PARAMETER GOLF — ABLATION SUITE (single GPU, 5-min runs)"
log "Configs  : ${#ABLATION_CONFIGS[@]}"
log "Budget   : ${T_BUDGET}s (3 hours hard cap, \$25 spend limit)"
log "Cloud    : ${CLOUD_TYPE}"
log "SSH key  : ${LOCAL_SSH_KEY}"
log "Artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "========================================================"

# Pre-flight checks
command -v curl >/dev/null 2>&1 || die "curl not found"
command -v jq   >/dev/null 2>&1 || die "jq not found"
command -v ssh  >/dev/null 2>&1 || die "ssh not found"
command -v scp  >/dev/null 2>&1 || die "scp not found"
[[ -f "$LOCAL_SSH_KEY" ]]    || die "SSH private key not found: $LOCAL_SSH_KEY"
[[ -f "$LOCAL_SSH_PUB" ]]    || die "SSH public key not found: $LOCAL_SSH_PUB"
[[ -f "${SCRIPT_DIR}/train_gpt.py" ]] || die "train_gpt.py not found in $SCRIPT_DIR"
[[ -f "${SCRIPT_DIR}/run_single_ablation.sh" ]] || die "run_single_ablation.sh not found in $SCRIPT_DIR"

# ── 1. Create pod ─────────────────────────────────────────────────────────────
POD_ID=""
create_pod

# Start budget clock
T_START=$(date +%s)
log "BUDGET CLOCK STARTED"

# ── 2. Wait for SSH ───────────────────────────────────────────────────────────
POD_HOST="" POD_PORT=""
wait_for_pod

# ── 3. GPU verification ──────────────────────────────────────────────────────
log "=== PHASE: Verification ==="
r "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader" | tee -a "$ORCH_LOG" || die "nvidia-smi failed"
r "python3 -c 'import torch; print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"gpus\",torch.cuda.device_count())'" \
    | tee -a "$ORCH_LOG" || die "PyTorch unavailable"

# ── 4. Install dependencies ──────────────────────────────────────────────────
log "=== PHASE: Install dependencies ==="
r "pip install -q sentencepiece zstandard && echo 'pip OK'" | tee -a "$ORCH_LOG" || die "pip install failed"
r "pip install -q huggingface-hub && echo 'hf hub OK'" | tee -a "$ORCH_LOG" || die "hf hub install failed"

# ── 5. Download data ──────────────────────────────────────────────────────────
log "=== PHASE: Download data (${DATA_SHARDS} train shards + val + tokenizer) ==="

r "mkdir -p /workspace/data/datasets/fineweb10B_sp1024 /workspace/data/tokenizers"

# Download val shard
log "Downloading val shard..."
r "python3 -c \"
from huggingface_hub import hf_hub_download
import shutil, os, pathlib
dest = '/workspace/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'
if os.path.exists(dest):
    print('val shard already exists')
else:
    p = hf_hub_download(repo_id='${HF_REPO}', filename='fineweb_val_000000.bin',
        subfolder='datasets/datasets/fineweb10B_sp1024', repo_type='dataset')
    src = pathlib.Path(p).resolve()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try: os.link(src, dest)
    except: shutil.copy2(src, dest)
    print('val shard downloaded')
\"" | tee -a "$ORCH_LOG" || die "Val shard download failed"

# Download tokenizer
log "Downloading tokenizer..."
r "python3 -c \"
from huggingface_hub import hf_hub_download
import shutil, os, pathlib
dest = '/workspace/data/tokenizers/fineweb_1024_bpe.model'
if os.path.exists(dest):
    print('tokenizer already exists')
else:
    p = hf_hub_download(repo_id='${HF_REPO}', filename='fineweb_1024_bpe.model',
        subfolder='datasets/tokenizers', repo_type='dataset')
    src = pathlib.Path(p).resolve()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try: os.link(src, dest)
    except: shutil.copy2(src, dest)
    print('tokenizer downloaded')
\"" | tee -a "$ORCH_LOG" || die "Tokenizer download failed"

# Download train shards
log "Downloading ${DATA_SHARDS} train shards..."
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
        return f'{fname}: exists'
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
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(download_shard, s): s for s in shards}
    for f in as_completed(futs):
        completed += 1
        result = f.result()
        if completed % 5 == 0 or completed == len(shards):
            print(f'  {completed}/{len(shards)} shards done', flush=True)

actual = len([f for f in os.listdir(dest_dir) if f.startswith('fineweb_train_')])
print(f'Download complete: {actual}/{len(shards)} shards')
if actual < 3:
    sys.exit(1)
PYEOF" | tee -a "$ORCH_LOG" || log "WARNING: Some train shards may have failed"

# Verify
TRAIN_COUNT=$(r "ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l")
VAL_COUNT=$(r "ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l")
log "Data verified: train=${TRAIN_COUNT} val=${VAL_COUNT}  T+$(t_elapsed)s"
[[ "$TRAIN_COUNT" -gt 0 ]] || die "No train shards found"
[[ "$VAL_COUNT"   -gt 0 ]] || die "No val shards found"

# ── 6. Upload code ────────────────────────────────────────────────────────────
log "=== PHASE: Upload code ==="
ul "${SCRIPT_DIR}/train_gpt.py" "${SCRIPT_DIR}/run_single_ablation.sh" || die "Code upload failed"
r "chmod +x /workspace/run_single_ablation.sh && mkdir -p /workspace/logs"
log "Code uploaded  T+$(t_elapsed)s"

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
log "=== PHASE: Smoke test (3 steps) ==="
SMOKE_LOG="${LOCAL_ARTIFACTS_DIR}/smoke_test.log"
r "cd /workspace && \
    MAX_WALLCLOCK_SECONDS=60 ITERATIONS=3 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 \
    CURRICULUM_ENABLED=0 SLIDING_EVAL=0 TEMP_SCALING=0 NGRAM_CACHE_ENABLED=0 \
    TURBO_QUANT_EXPORT=0 HESSIAN_TERNARY_GPTQ=0 SELECTIVE_PRUNING=0 GPTQ_LITE_ENABLED=0 \
    ARCHITECTURE=skc NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4 \
    VOCAB_SIZE=1024 TRAIN_BATCH_TOKENS=32768 TRAIN_SEQ_LEN=1024 COMPILE_MODE=none SEED=42 \
    DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
    OMP_NUM_THREADS=1 timeout 120 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1" \
    > "$SMOKE_LOG" 2>&1 || log "WARNING: smoke test non-zero exit"

SMOKE_ERR=$(grep -cE "^Traceback|CUDA error|RuntimeError|ImportError|OutOfMemoryError" "$SMOKE_LOG" 2>/dev/null || echo 0)
SMOKE_OK=$(grep -cE "step:[0-9]" "$SMOKE_LOG" 2>/dev/null || echo 0)
SMOKE_NAN=$(grep -cE "loss:nan" "$SMOKE_LOG" 2>/dev/null || echo 0)

if [[ "$SMOKE_ERR" -gt 0 ]]; then
    log "Smoke test FAILED — errors detected. Checking OOM..."
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$SMOKE_LOG" 2>/dev/null; then
        log "OOM detected — reducing batch to 16384"
        r "sed -i 's/TRAIN_BATCH_TOKENS=32768/TRAIN_BATCH_TOKENS=16384/' /workspace/run_single_ablation.sh"
        # Retry smoke
        r "cd /workspace && \
            MAX_WALLCLOCK_SECONDS=60 ITERATIONS=3 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 \
            CURRICULUM_ENABLED=0 SLIDING_EVAL=0 TEMP_SCALING=0 NGRAM_CACHE_ENABLED=0 \
            TURBO_QUANT_EXPORT=0 HESSIAN_TERNARY_GPTQ=0 SELECTIVE_PRUNING=0 GPTQ_LITE_ENABLED=0 \
            ARCHITECTURE=skc NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4 \
            VOCAB_SIZE=1024 TRAIN_BATCH_TOKENS=16384 TRAIN_SEQ_LEN=1024 COMPILE_MODE=none SEED=42 \
            DATA_PATH=/workspace/data/datasets/fineweb10B_sp1024 \
            TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
            OMP_NUM_THREADS=1 timeout 120 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1" \
            > "${SMOKE_LOG}.retry" 2>&1 || true
        SMOKE_OK=$(grep -cE "step:[0-9]" "${SMOKE_LOG}.retry" 2>/dev/null || echo 0)
        [[ "$SMOKE_OK" -eq 0 ]] && die "Smoke test failed even with reduced batch"
        log "Smoke test PASSED with reduced batch ✓"
    else
        cat "$SMOKE_LOG" | tail -30 | tee -a "$ORCH_LOG"
        die "Smoke test failed with non-OOM errors. See smoke_test.log"
    fi
elif [[ "$SMOKE_NAN" -gt 0 ]]; then
    die "Smoke test: NaN loss detected"
elif [[ "$SMOKE_OK" -eq 0 ]]; then
    cat "$SMOKE_LOG" | tail -20 | tee -a "$ORCH_LOG"
    die "Smoke test: no training output"
else
    log "Smoke test PASSED ✓  T+$(t_elapsed)s"
fi

# ── 8. Run ablation suite ────────────────────────────────────────────────────
log "========================================================"
log "=== PHASE: Running ${#ABLATION_CONFIGS[@]} ablation configs ==="
log "========================================================"

RESULTS=()
COMPLETED=0
SKIPPED=0

for config in "${ABLATION_CONFIGS[@]}"; do
    run_ablation "$config"
    COMPLETED=$((COMPLETED + 1))
    log "Progress: ${COMPLETED}/${#ABLATION_CONFIGS[@]}  T+$(t_elapsed)s"
done

# ── 9. Download all logs ──────────────────────────────────────────────────────
log "=== PHASE: Downloading all logs ==="
for config in "${ABLATION_CONFIGS[@]}"; do
    download_from_pod "/workspace/logs/ablation_${config}.log" \
        "${LOCAL_ARTIFACTS_DIR}/ablation_${config}.log" 0 1 || true
done
log "Logs downloaded  T+$(t_elapsed)s"

# ── 10. Terminate pod ─────────────────────────────────────────────────────────
log "=== Terminating pod ${POD_ID} ==="
terminate_pod "$POD_ID"

# ── 11. Summary ──────────────────────────────────────────────────────────────
SUMMARY_FILE="${LOCAL_ARTIFACTS_DIR}/summary.md"
{
    echo "# Ablation Results — $(date)"
    echo ""
    echo "| # | Config | Val BPB | Status |"
    echo "|---|--------|---------|--------|"
    i=1
    for result in "${RESULTS[@]}"; do
        IFS='|' read -r cfg bpb status <<< "$result"
        echo "| $i | \`${cfg}\` | ${bpb} | ${status} |"
        i=$((i + 1))
    done
    echo ""
    echo "GPU: $(cat "${LOCAL_ARTIFACTS_DIR}/gpu_type.txt" 2>/dev/null || echo unknown)"
    echo "Pod: $(cat "${LOCAL_ARTIFACTS_DIR}/pod_id.txt" 2>/dev/null || echo unknown)"
    echo "Total time: $(t_elapsed)s"
} > "$SUMMARY_FILE"

log "========================================================"
log "ABLATION SUITE COMPLETE — $(t_elapsed)s total"
log "========================================================"
log ""
log "Results:"
printf "%-20s  %-12s  %s\n" "CONFIG" "VAL_BPB" "STATUS" | tee -a "$ORCH_LOG"
printf "%-20s  %-12s  %s\n" "------" "-------" "------" | tee -a "$ORCH_LOG"
for result in "${RESULTS[@]}"; do
    IFS='|' read -r cfg bpb status <<< "$result"
    printf "%-20s  %-12s  %s\n" "$cfg" "$bpb" "$status" | tee -a "$ORCH_LOG"
done
log ""
log "All artifacts: ${LOCAL_ARTIFACTS_DIR}"
log "Summary:       ${SUMMARY_FILE}"
log "Full log:      ${ORCH_LOG}"
