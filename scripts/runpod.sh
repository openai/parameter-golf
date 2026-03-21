#!/usr/bin/env bash
# RunPod automation for Parameter Golf
# Uses REST API v1 with curl — no dependencies beyond bash, curl, jq
#
# Usage:
#   source .env  # loads RUNPOD_API_KEY
#   ./scripts/runpod.sh create [1|8]          # create pod with 1 or 8 H100s
#   ./scripts/runpod.sh status                # show pod status
#   ./scripts/runpod.sh setup                 # clone repo + download data on pod
#   ./scripts/runpod.sh run [RUN_ID] [EXTRA]  # run training on pod
#   ./scripts/runpod.sh logs [RUN_ID]         # fetch logs from pod
#   ./scripts/runpod.sh fetch [RUN_ID]        # fetch all artifacts from pod
#   ./scripts/runpod.sh stop                  # stop pod (keeps volume)
#   ./scripts/runpod.sh terminate             # terminate pod (deletes everything)
#   ./scripts/runpod.sh list                  # list all pods

set -euo pipefail

# Load .env if it exists
ENV_FILE="${BASH_SOURCE[0]%/*}/../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

API_BASE="https://rest.runpod.io/v1"
API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY in .env or environment}"

# Pod state file — tracks the current pod ID
STATE_FILE="${BASH_SOURCE[0]%/*}/../.runpod_pod_id"

# Parameter Golf template from the challenge README
TEMPLATE_ID="y5cejece4j"
GITHUB_REPO="https://github.com/rarce/parameter-golf.git"
GITHUB_BRANCH="sota-review"

# ---- helpers ----

_api() {
    local method="$1" path="$2"
    shift 2
    curl -s --fail-with-body \
        -X "$method" \
        -H "Authorization: Bearer ${API_KEY}" \
        -H "Content-Type: application/json" \
        "${API_BASE}${path}" "$@"
}

_pod_id() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo "No pod tracked. Run: $0 create" >&2
        exit 1
    fi
}

_get_ssh_info() {
    local pod_id
    pod_id=$(_pod_id)
    local pod_info
    pod_info=$(_api GET "/pods/${pod_id}")
    local ip port
    ip=$(echo "$pod_info" | jq -r '.publicIp // empty')
    port=$(echo "$pod_info" | jq -r '.portMappings["22"] // empty')
    if [[ -z "$ip" || -z "$port" ]]; then
        echo "Pod SSH not ready yet. Check status." >&2
        exit 1
    fi
    echo "${ip} ${port}"
}

_ssh_cmd() {
    local info ip port
    info=$(_get_ssh_info)
    ip=$(echo "$info" | awk '{print $1}')
    port=$(echo "$info" | awk '{print $2}')
    echo "ssh -o StrictHostKeyChecking=no -p ${port} root@${ip}"
}

_scp_cmd() {
    local info ip port
    info=$(_get_ssh_info)
    ip=$(echo "$info" | awk '{print $1}')
    port=$(echo "$info" | awk '{print $2}')
    echo "scp -o StrictHostKeyChecking=no -P ${port} root@${ip}"
}

# ---- commands ----

cmd_create() {
    local gpu_count="${1:-1}"
    local gpu_type="NVIDIA H100 80GB HBM3"
    local name="pgolf-${gpu_count}xh100"

    echo "Creating pod: ${name} (${gpu_count}x H100)..."
    local result
    result=$(_api POST "/pods" -d "{
        \"name\": \"${name}\",
        \"gpuTypeIds\": [\"${gpu_type}\"],
        \"gpuCount\": ${gpu_count},
        \"templateId\": \"${TEMPLATE_ID}\",
        \"containerDiskInGb\": 50,
        \"volumeInGb\": 50,
        \"volumeMountPath\": \"/workspace\",
        \"ports\": [\"22/tcp\", \"8888/http\"]
    }")

    local pod_id
    pod_id=$(echo "$result" | jq -r '.id // .pod.id // empty')
    if [[ -z "$pod_id" ]]; then
        echo "Failed to create pod:" >&2
        echo "$result" | jq . 2>/dev/null || echo "$result"
        exit 1
    fi

    echo "$pod_id" > "$STATE_FILE"
    echo "Pod created: ${pod_id}"
    echo "Waiting for pod to be ready..."
    sleep 10
    cmd_status
}

cmd_list() {
    _api GET "/pods" | jq -r '.[] | "\(.id)\t\(.name)\t\(.desiredStatus)\t\(.costPerHr)$/hr"' 2>/dev/null || \
    _api GET "/pods" | jq .
}

cmd_status() {
    local pod_id
    pod_id=$(_pod_id)
    local info
    info=$(_api GET "/pods/${pod_id}")
    echo "$info" | jq '{
        id: .id,
        name: .name,
        status: .desiredStatus,
        gpu: .machine.gpuTypeId,
        gpu_count: .gpuCount,
        cost_per_hr: .costPerHr,
        ssh: "\(.publicIp):\(.portMappings["22"])",
        image: .imageName
    }' 2>/dev/null || echo "$info" | jq .
}

cmd_ssh() {
    local ssh_cmd
    ssh_cmd=$(_ssh_cmd)
    echo "Connecting: ${ssh_cmd}"
    eval "$ssh_cmd"
}

cmd_setup() {
    local ssh_cmd
    ssh_cmd=$(_ssh_cmd)
    echo "Setting up pod..."
    eval "$ssh_cmd" << SETUP_EOF
set -ex
cd /workspace
if [ -d parameter-golf ] && [ ! -d parameter-golf/.git ]; then
    rm -rf parameter-golf  # template creates empty dir
fi
if [ ! -d parameter-golf ]; then
    git clone -b ${GITHUB_BRANCH} ${GITHUB_REPO}
else
    cd parameter-golf && git fetch origin && git checkout ${GITHUB_BRANCH} && git pull origin ${GITHUB_BRANCH} && cd ..
fi
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "Setup complete. Data downloaded."
SETUP_EOF
}

cmd_run() {
    local run_id="${1:-consensus_v1}"
    shift || true
    local extra_env="${*:-}"
    local ssh_cmd
    ssh_cmd=$(_ssh_cmd)

    local gpu_count
    gpu_count=$(_api GET "/pods/$(_pod_id)" | jq -r '.gpuCount // 1')

    echo "Starting training: RUN_ID=${run_id} on ${gpu_count}x GPU..."
    eval "$ssh_cmd" << RUN_EOF
cd /workspace/parameter-golf
mkdir -p logs
RUN_ID=${run_id} \\
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \\
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
VOCAB_SIZE=1024 \\
${extra_env} \\
torchrun --standalone --nproc_per_node=${gpu_count} train_gpt.py 2>&1 | tee logs/${run_id}.txt
RUN_EOF
}

cmd_logs() {
    local run_id="${1:-consensus_v1}"
    local ssh_cmd
    ssh_cmd=$(_ssh_cmd)
    eval "$ssh_cmd" "tail -50 /workspace/parameter-golf/logs/${run_id}.txt" 2>/dev/null || \
    eval "$ssh_cmd" "cat /workspace/parameter-golf/logs/${run_id}.txt"
}

cmd_fetch() {
    local run_id="${1:-consensus_v1}"
    local local_dir="logs/runpod_${run_id}"
    mkdir -p "$local_dir"

    local scp_base
    scp_base=$(_scp_cmd)

    echo "Fetching artifacts for ${run_id}..."
    eval "${scp_base}:/workspace/parameter-golf/logs/${run_id}.txt ${local_dir}/" 2>/dev/null || true
    eval "${scp_base}:/workspace/parameter-golf/final_model.int8.ptz ${local_dir}/" 2>/dev/null || true
    eval "${scp_base}:/workspace/parameter-golf/final_model.pt ${local_dir}/" 2>/dev/null || true
    echo "Artifacts saved to ${local_dir}/"
    ls -lh "$local_dir/"
}

cmd_stop() {
    local pod_id
    pod_id=$(_pod_id)
    echo "Stopping pod ${pod_id} (volume preserved)..."
    _api POST "/pods/${pod_id}/stop" | jq .
}

cmd_terminate() {
    local pod_id
    pod_id=$(_pod_id)
    echo "Terminating pod ${pod_id} (everything deleted)..."
    _api DELETE "/pods/${pod_id}" | jq .
    rm -f "$STATE_FILE"
    echo "Pod terminated."
}

# ---- main ----

case "${1:-help}" in
    create)    cmd_create "${2:-1}" ;;
    list)      cmd_list ;;
    status)    cmd_status ;;
    ssh)       cmd_ssh ;;
    setup)     cmd_setup ;;
    run)       shift; cmd_run "$@" ;;
    logs)      cmd_logs "${2:-consensus_v1}" ;;
    fetch)     cmd_fetch "${2:-consensus_v1}" ;;
    stop)      cmd_stop ;;
    terminate) cmd_terminate ;;
    *)
        echo "Usage: $0 {create|list|status|ssh|setup|run|logs|fetch|stop|terminate}"
        echo ""
        echo "Workflow:"
        echo "  source .env"
        echo "  $0 create 1          # 1x H100 (~\$3/hr)"
        echo "  $0 setup              # clone repo + download data"
        echo "  $0 run consensus_v1   # start training"
        echo "  $0 fetch consensus_v1 # get results locally"
        echo "  $0 stop               # stop (keep volume)"
        echo "  $0 terminate          # delete everything"
        ;;
esac
