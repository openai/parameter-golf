#!/usr/bin/env bash
# =============================================================================
# 4-Experiment 2-GPU Sweep
# Runs 4 sequential 10-minute experiments on a freshly-provisioned 2-GPU pod,
# terminating all stale pods first. Results inform the final 8xH100 submission.
#
# Experiments:
#   Exp1: 16 experts, 256D, batch=16384  (scale MoE experts)
#   Exp2: 32 experts, 256D, batch=16384  (push expert ceiling)
#   Exp3:  8 experts, 256D, batch=8192   (more steps, smaller batch)
#   Exp4:  8 experts, 320D, batch=16384  (wider model)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SWEEP_LOG="${SCRIPT_DIR}/sweep_4exp_2gpu_$(date +%Y%m%d_%H%M%S).log"
log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*" | tee -a "$SWEEP_LOG"; }

export RUNPOD_API_KEY="${RUNPOD_API_KEY:?ERROR: Please export RUNPOD_API_KEY before running.}"

# ---------------------------------------------------------------------------
# Step 1: Terminate ALL existing pods
# ---------------------------------------------------------------------------
log "=== Terminating all live pods ==="
RUNPOD_API="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

list_all_pods() {
    curl -s --request POST \
        --header "content-type: application/json" \
        --url "$RUNPOD_API" \
        --data "$(jq -n --arg q 'query { myself { pods { id name desiredStatus } } }' '{query: $q}')" \
    | jq -r '.data.myself.pods[]? | select(.desiredStatus == "RUNNING") | .id'
}

terminate_pod_by_id() {
    local pid="$1"
    curl -s --request POST \
        --header "content-type: application/json" \
        --url "$RUNPOD_API" \
        --data "$(jq -n --arg q "mutation { podTerminate(input: { podId: \"${pid}\" }) }" '{query: $q}')" \
        | jq -r '.data // .errors' >/dev/null
    log "  Terminated pod: ${pid}"
}

while IFS= read -r pid; do
    [[ -z "${pid:-}" ]] && continue
    terminate_pod_by_id "$pid"
done < <(list_all_pods)
log "All pods terminated. Waiting 30s for cleanup..."
sleep 30

# ---------------------------------------------------------------------------
# Step 2: Common env for all experiments
# ---------------------------------------------------------------------------
export GPU_COUNT=2
export MIN_GPU_MEMORY_GB=20   # 20GB+ VRAM per GPU (A40/3090/A100 class, avoid H100 cost)
export DATA_SHARDS=2
export SEED=1337
export EXPORT_ONLY=0
export KEEP_POD_ON_EXIT=1   # keep pod alive between experiments
export TERNARY_THRESHOLD_SEARCH=1  # enabled: eval throughput fix prevents the multi-hour export hang
export SLIDING_EVAL=0              # disabled: 62M token sliding eval takes hours, not needed for sweeps
unset POD_ID   # force fresh pod creation on first run
unset TRAIN_BATCH_TOKENS  # let the pod detect VRAM for max parallelism
unset VAL_BATCH_SIZE

# NCCL: RunPod PCIe multi-GPU needs P2P disabled to avoid 600s topology-probe hang
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo

# ---------------------------------------------------------------------------
# Helper: run one experiment
# ---------------------------------------------------------------------------
run_experiment() {
    local exp_name="$1"; shift
    # remaining args are env var assignments: KEY=VALUE ...
    local exp_log="${SCRIPT_DIR}/exp_${exp_name}_$(date +%Y%m%d_%H%M%S).log"

    log "======================================================"
    log "Starting ${exp_name}"
    log "Config: $*"
    log "Log: ${exp_log}"
    log "======================================================"

    # Export each KEY=VALUE pair
    for kv in "$@"; do
        export "${kv?}"
    done

    bash "${SCRIPT_DIR}/orchestrate_small_skc_multigpu_runpod.sh" 2>&1 | tee "$exp_log"

    # Extract the best BPB from the log
    local best_bpb
    best_bpb=$(grep -oP 'val_bpb:\K[0-9.]+' "$exp_log" | sort -n | head -1 || echo "N/A")
    local final_bpb
    final_bpb=$(grep 'final_ternary_roundtrip' "$exp_log" | grep -oP 'val_bpb:\K[0-9.]+' | tail -1 || echo "N/A")
    local artifact_mb
    artifact_mb=$(grep 'artifact:' "$exp_log" | grep -oP 'artifact:\K[0-9.]+MB' | tail -1 || echo "N/A")

    log "--- ${exp_name} RESULTS: final_bpb=${final_bpb} best_proxy_bpb=${best_bpb} artifact=${artifact_mb} ---"

    # Capture POD_ID so next experiment reuses the same pod
    local pod_line
    pod_line=$(grep -oP 'success: pod=\K[a-z0-9]+' "$exp_log" | head -1 || true)
    # Also try the "Re-using existing pod" line
    [[ -z "${pod_line:-}" ]] && pod_line=$(grep -oP 'Re-using existing pod: \K[a-z0-9]+' "$exp_log" | head -1 || true)
    if [[ -n "${pod_line:-}" && "${pod_line}" != "null" ]]; then
        export POD_ID="$pod_line"
    fi
}

# ---------------------------------------------------------------------------
# Experiment 1: Scale MoE to 16 experts
# ---------------------------------------------------------------------------
run_experiment "exp1_16experts_256d" \
    "NUM_LAYERS=8" \
    "MODEL_DIM=256" \
    "KOOPMAN_MIXER_RANK=4" \
    "MOE_ENABLED=1" \
    "MOE_NUM_EXPERTS=16" \
    "MOE_TOP_K=2" \
    "BIGRAM_HASH_BUCKETS=16384"

# ---------------------------------------------------------------------------
# Experiment 2: Push to 32 experts (near budget ceiling)
# ---------------------------------------------------------------------------
run_experiment "exp2_32experts_256d" \
    "NUM_LAYERS=8" \
    "MODEL_DIM=256" \
    "KOOPMAN_MIXER_RANK=4" \
    "MOE_ENABLED=1" \
    "MOE_NUM_EXPERTS=32" \
    "MOE_TOP_K=4" \
    "BIGRAM_HASH_BUCKETS=16384"

# ---------------------------------------------------------------------------
# Experiment 3: Smaller batch → more steps (8 experts, batch=8192)
# ---------------------------------------------------------------------------
run_experiment "exp3_8experts_256d" \
    "NUM_LAYERS=8" \
    "MODEL_DIM=256" \
    "KOOPMAN_MIXER_RANK=4" \
    "MOE_ENABLED=1" \
    "MOE_NUM_EXPERTS=8" \
    "MOE_TOP_K=4" \
    "BIGRAM_HASH_BUCKETS=16384"

# ---------------------------------------------------------------------------
# Experiment 4: Wider model (320D, 8 experts)
# ---------------------------------------------------------------------------
run_experiment "exp4_8experts_320d" \
    "NUM_LAYERS=8" \
    "MODEL_DIM=320" \
    "KOOPMAN_MIXER_RANK=4" \
    "MOE_ENABLED=1" \
    "MOE_NUM_EXPERTS=8" \
    "MOE_TOP_K=4" \
    "BIGRAM_HASH_BUCKETS=16384"

# ---------------------------------------------------------------------------
# Terminate pod when done (all 4 experiments complete)
# ---------------------------------------------------------------------------
log "=== All 4 experiments complete. Terminating pod ${POD_ID:-unknown} ==="
export KEEP_POD_ON_EXIT=0
[[ -n "${POD_ID:-}" ]] && terminate_pod_by_id "$POD_ID"

log "=== SWEEP COMPLETE. See ${SWEEP_LOG} for summary ==="
