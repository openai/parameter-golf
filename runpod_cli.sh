#!/bin/bash
# ============================================================================
# Parameter Golf — RunPod CLI Deployment
# ============================================================================
#
# ARCHITECTURE:
#   Network volume (US-GA-2) holds data persistently across pods.
#   Create/delete pods freely — volume survives.
#   Cannot change GPU count on existing pod — must delete and recreate.
#
# BUDGET ($25):
#   Network volume 50GB:     $0.07/GB/month = $3.50/month
#   8x H100 on-demand:       $21.52/hr = $5.38 per 15-min run
#   8x H100 spot:            $12.00/hr = $3.00 per 15-min run (risk of interruption)
#   Budget for 3 on-demand:  $3.50 + $16.14 = ~$20
#   Budget for 5 spot:       $3.50 + $15.00 = ~$19
#
# GOTCHAS (from deep research):
#   - Network volumes require cloudType=SECURE
#   - Network volume locks pod to same datacenter
#   - Pod volumes (volumeInGb) are NOT network volumes — data dies with pod
#   - Cannot change GPU count on existing pod — delete and recreate
#   - Stopped pods charge $0.20/GB/month for pod volumes (2x running rate)
#   - Network volumes charge $0.07/GB/month regardless of state
#   - Always DELETE pods when done, don't just stop them
#
# ============================================================================

set -e

# --- Configuration ---
RUNPOD_API_KEY=$(grep -oP "apikey = '\K[^']+" ~/.runpod/config.toml 2>/dev/null || echo "")
NETWORK_VOLUME_ID="qwervlzocy"
DATACENTER="US-GA-2"
GPU_TYPE="NVIDIA H100 80GB HBM3"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
SCRIPT_PATH="experiments/r3/exp11_vrl_xsa11_noema.py"

# State file
STATE_FILE="$HOME/.runpod_golf_state"
save_state() { echo "$1=$2" >> "$STATE_FILE"; }
load_state() { [ -f "$STATE_FILE" ] && source "$STATE_FILE" || true; }

api() {
    curl -s --request "$1" \
        --url "https://rest.runpod.io/v1$2" \
        --header "Authorization: Bearer $RUNPOD_API_KEY" \
        --header "Content-Type: application/json" \
        ${3:+--data "$3"}
}

graphql() {
    curl -s --request POST \
        --url "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        --header "Content-Type: application/json" \
        --data "{\"query\": \"$1\"}"
}

# ============================================================================
# Preflight
# ============================================================================
check() {
    echo "=== Preflight ==="
    local ok=true

    if ! command -v runpodctl &>/dev/null; then
        echo "[FAIL] runpodctl not installed"
        ok=false
    else
        echo "[OK] runpodctl $(runpodctl version 2>/dev/null)"
    fi

    if [ -z "$RUNPOD_API_KEY" ]; then
        echo "[FAIL] No API key in ~/.runpod/config.toml"
        ok=false
    else
        echo "[OK] API key configured"
    fi

    [ -f "$SSH_KEY" ] && echo "[OK] SSH key: $SSH_KEY" || echo "[WARN] No SSH key"
    [ -f "$SCRIPT_PATH" ] && echo "[OK] Script: $SCRIPT_PATH ($(wc -c < "$SCRIPT_PATH") bytes)" || echo "[FAIL] Script missing: $SCRIPT_PATH"

    echo ""
    echo "=== Network Volume ==="
    api GET "/networkvolumes" | python3 -c "
import json, sys
vols = json.load(sys.stdin)
if not vols:
    print('[WARN] No network volumes. Run: ./runpod_cli.sh create-volume')
for v in vols:
    print(f'[OK] {v[\"id\"]}  {v[\"name\"]}  {v[\"size\"]}GB  DC:{v[\"dataCenterId\"]}')
" 2>&1

    echo ""
    echo "=== GPU Availability (8x H100 SXM) ==="
    graphql '{ gpuTypes(input: { id: \"NVIDIA H100 80GB HBM3\" }) { displayName lowestPrice(input: { gpuCount: 8 }) { stockStatus uninterruptablePrice minimumBidPrice } } }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
gpu = d['data']['gpuTypes'][0]
lp = gpu['lowestPrice']
print(f'Stock: {lp[\"stockStatus\"]}  On-demand: \${lp[\"uninterruptablePrice\"]}/hr  Spot min: \${lp[\"minimumBidPrice\"]}/hr')
" 2>&1

    echo ""
    echo "=== Pods ==="
    runpodctl get pod 2>&1 | grep -v "deprecated" || echo "No pods"

    $ok && echo -e "\nReady." || echo -e "\nFix issues above."
}

# ============================================================================
# Check datacenter H100 availability
# ============================================================================
availability() {
    echo "=== H100 SXM Availability by Datacenter ==="
    graphql '{ dataCenters { id gpuAvailability { gpuTypeId stockStatus } } }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = []
for dc in data['data']['dataCenters']:
    for gpu in dc.get('gpuAvailability',[]):
        if 'H100' in gpu.get('gpuTypeId','') and '80GB' in gpu.get('gpuTypeId',''):
            results.append((dc['id'], gpu.get('stockStatus','None')))
results.sort(key=lambda x: {'High':0,'Medium':1,'Low':2,'None':3}.get(x[1],4))
print(f'{\"DC\":15s} {\"Stock\":10s}')
print('-' * 28)
for dc, stock in results:
    marker = ' <-- our volume' if dc == 'US-GA-2' else ''
    print(f'{dc:15s} {stock:10s}{marker}')
" 2>&1

    echo ""
    echo "=== 8x H100 Global Stock ==="
    graphql '{ gpuTypes(input: { id: \"NVIDIA H100 80GB HBM3\" }) { lowestPrice(input: { gpuCount: 8 }) { stockStatus uninterruptablePrice minimumBidPrice } } }' | python3 -c "
import json, sys
lp = json.load(sys.stdin)['data']['gpuTypes'][0]['lowestPrice']
print(f'8x Stock: {lp[\"stockStatus\"]}  On-demand: \${lp[\"uninterruptablePrice\"]}/hr  Spot: \${lp[\"minimumBidPrice\"]}/hr')
" 2>&1
}

# ============================================================================
# Create 8x H100 training pod with network volume
# ============================================================================
train() {
    echo "=== Create 8x H100 Training Pod ==="
    echo "Network Volume: $NETWORK_VOLUME_ID (DC: $DATACENTER)"
    echo "Cost: ~\$21.52/hr on-demand, ~\$12/hr spot"
    echo ""

    # Check availability first
    echo "Checking 8x H100 stock..."
    local stock=$(graphql '{ gpuTypes(input: { id: \"NVIDIA H100 80GB HBM3\" }) { lowestPrice(input: { gpuCount: 8 }) { stockStatus } } }' | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['gpuTypes'][0]['lowestPrice']['stockStatus'])" 2>&1)
    echo "8x H100 stock: $stock"

    if [ "$stock" = "None" ]; then
        echo "ERROR: No 8x H100 available right now. Try again later."
        exit 1
    fi

    read -p "Create pod? [y/N]: " confirm
    [ "$confirm" != "y" ] && exit 0

    echo "Creating pod..."
    local result=$(api POST "/pods" "{
        \"name\": \"pg-train\",
        \"imageName\": \"$IMAGE\",
        \"gpuTypeIds\": [\"$GPU_TYPE\"],
        \"gpuCount\": 8,
        \"networkVolumeId\": \"$NETWORK_VOLUME_ID\",
        \"volumeMountPath\": \"/workspace\",
        \"containerDiskInGb\": 20,
        \"cloudType\": \"SECURE\",
        \"ports\": [\"22/tcp\"],
        \"supportPublicIp\": true
    }")

    echo "$result" | python3 -m json.tool 2>/dev/null || echo "$result"

    local pod_id=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)
    if [ -n "$pod_id" ] && [ "$pod_id" != "" ]; then
        echo "TRAIN_POD_ID=$pod_id" > "$STATE_FILE"
        echo ""
        echo "Pod ID: $pod_id"
        echo ""
        echo "=== Next Steps ==="
        echo "1. Wait for ready:     ./runpod_cli.sh wait"
        echo "2. Upload script:      ./runpod_cli.sh upload"
        echo "3. Run training:       ./runpod_cli.sh run"
        echo "4. Download results:   ./runpod_cli.sh download"
        echo "5. DELETE pod:         ./runpod_cli.sh delete"
    else
        echo "ERROR: Pod creation failed. Check output above."
        exit 1
    fi
}

# ============================================================================
# Wait for pod to be ready and show SSH info
# ============================================================================
wait_ready() {
    load_state
    if [ -z "$TRAIN_POD_ID" ]; then
        echo "No pod ID. Run: ./runpod_cli.sh train"
        exit 1
    fi

    echo "Waiting for pod $TRAIN_POD_ID..."
    for i in $(seq 1 30); do
        local info=$(api GET "/pods/$TRAIN_POD_ID")
        local ip=$(echo "$info" | python3 -c "import json,sys; print(json.load(sys.stdin).get('publicIp',''))" 2>/dev/null)
        local port=$(echo "$info" | python3 -c "import json,sys; print(json.load(sys.stdin).get('portMappings',{}).get('22',''))" 2>/dev/null)

        if [ -n "$ip" ] && [ "$ip" != "" ] && [ -n "$port" ] && [ "$port" != "" ]; then
            echo ""
            echo "=== Pod Ready ==="
            echo "SSH: ssh -i $SSH_KEY -p $port root@$ip"
            echo "IP: $ip  Port: $port"
            save_state "POD_IP" "$ip"
            save_state "POD_PORT" "$port"
            return 0
        fi
        echo "  ... waiting ($i/30)"
        sleep 10
    done
    echo "Timed out. Check: runpodctl pod list"
    exit 1
}

# ============================================================================
# Upload submission script to pod
# ============================================================================
upload() {
    load_state
    if [ -z "$POD_IP" ] || [ -z "$POD_PORT" ]; then
        echo "No pod connection info. Run: ./runpod_cli.sh wait"
        exit 1
    fi

    echo "Uploading $SCRIPT_PATH to pod as train_submission.py..."
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$POD_PORT" \
        "$SCRIPT_PATH" "root@$POD_IP:/workspace/parameter-golf/train_submission.py"
    echo "Done."
}

# ============================================================================
# Run training on the pod
# ============================================================================
run() {
    load_state
    if [ -z "$POD_IP" ] || [ -z "$POD_PORT" ]; then
        echo "No pod connection info. Run: ./runpod_cli.sh wait"
        exit 1
    fi

    echo "=== Starting training on 8x H100 ==="
    echo "This will take ~10-15 minutes. Do NOT close this terminal."
    echo ""

    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$POD_PORT" "root@$POD_IP" bash -s << 'TRAINING'
set -e
cd /workspace/parameter-golf

echo "=== Environment ==="
echo "GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "Shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
echo "Network volume: $(df -h /workspace | tail -1 | awk '{print $1}')"

# Install deps if needed
pip install -q sentencepiece 2>/dev/null || true

# Verify submission script exists
if [ ! -f train_submission.py ]; then
    echo "ERROR: train_submission.py not found. Run upload first."
    exit 1
fi

RUN_ID="submission_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "=== TRAINING START (RUN_ID: $RUN_ID) ==="
echo "Wallclock limit: 600s (10 min)"
echo ""

MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_submission.py 2>&1 | tee "/workspace/run_${RUN_ID}.log"

echo ""
echo "=========================================="
echo "=== RESULTS ==="
echo "=========================================="
grep "val_bpb" "/workspace/run_${RUN_ID}.log" | tail -5 || echo "No val_bpb found"
echo ""

# Artifact size check
ARTIFACT=$(ls -1 final_model.*.ptz 2>/dev/null | head -1)
if [ -n "$ARTIFACT" ]; then
    ASIZE=$(stat -c%s "$ARTIFACT")
    CSIZE=$(wc -c < train_submission.py)
    TOTAL=$((ASIZE + CSIZE))
    echo "Artifact:  $ASIZE bytes"
    echo "Code:      $CSIZE bytes"
    echo "Total:     $TOTAL / 16,000,000 bytes"
    if [ "$TOTAL" -le 16000000 ]; then
        echo "STATUS: FITS under 16MB"
    else
        echo "STATUS: OVER by $((TOTAL - 16000000)) bytes"
    fi
    # Copy artifact to network volume for persistence
    cp "$ARTIFACT" "/workspace/" 2>/dev/null || true
    cp train_submission.py "/workspace/" 2>/dev/null || true
else
    echo "No artifact found — check log for errors"
fi

echo ""
echo "=== DELETE THE POD NOW: ./runpod_cli.sh delete ==="
TRAINING
}

# ============================================================================
# Download results from pod
# ============================================================================
download() {
    load_state
    if [ -z "$POD_IP" ] || [ -z "$POD_PORT" ]; then
        echo "No pod connection info. Run: ./runpod_cli.sh wait"
        exit 1
    fi

    echo "Downloading results..."
    mkdir -p ./runpod_results

    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$POD_PORT" \
        "root@$POD_IP:/workspace/parameter-golf/final_model.*.ptz" ./runpod_results/ 2>/dev/null || echo "No .ptz artifacts"

    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$POD_PORT" \
        "root@$POD_IP:/workspace/run_*.log" ./runpod_results/ 2>/dev/null || echo "No logs"

    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$POD_PORT" \
        "root@$POD_IP:/workspace/parameter-golf/train_submission.py" ./runpod_results/ 2>/dev/null || echo "No script"

    echo ""
    echo "Results in ./runpod_results/:"
    ls -lh ./runpod_results/ 2>/dev/null || echo "(empty)"
    echo ""
    echo "=== DELETE THE POD NOW: ./runpod_cli.sh delete ==="
}

# ============================================================================
# Delete training pod (STOP BILLING)
# ============================================================================
delete() {
    load_state
    if [ -z "$TRAIN_POD_ID" ]; then
        echo "No pod to delete."
        runpodctl get pod 2>&1 | grep -v deprecated
        exit 0
    fi

    echo "Deleting pod $TRAIN_POD_ID..."
    api DELETE "/pods/$TRAIN_POD_ID"
    echo "Pod deleted. Network volume qwervlzocy persists with data."
    rm -f "$STATE_FILE"
}

# ============================================================================
# Status
# ============================================================================
status() {
    echo "=== Pods ==="
    runpodctl get pod 2>&1 | grep -v deprecated

    echo ""
    echo "=== Network Volumes ==="
    api GET "/networkvolumes" | python3 -c "
import json, sys
for v in json.load(sys.stdin):
    print(f'  {v[\"id\"]}  {v[\"name\"]}  {v[\"size\"]}GB  DC:{v[\"dataCenterId\"]}')
" 2>&1

    load_state
    echo ""
    [ -n "$TRAIN_POD_ID" ] && echo "Active pod: $TRAIN_POD_ID"
    [ -n "$POD_IP" ] && echo "SSH: ssh -i $SSH_KEY -p $POD_PORT root@$POD_IP"
}

# ============================================================================
# Full cleanup (end of competition)
# ============================================================================
cleanup() {
    echo "=== Full Cleanup ==="
    echo "This deletes ALL pods and the network volume."
    echo "Data on the volume will be PERMANENTLY LOST."
    read -p "Continue? [y/N]: " confirm
    [ "$confirm" != "y" ] && exit 0

    # Delete all pods
    runpodctl get pod 2>&1 | grep -v "deprecated\|ID\|^$" | awk '{print $1}' | while read pid; do
        [ -n "$pid" ] && api DELETE "/pods/$pid" && echo "Deleted pod $pid"
    done

    # Delete network volume
    api DELETE "/networkvolumes/$NETWORK_VOLUME_ID" && echo "Deleted volume $NETWORK_VOLUME_ID"
    rm -f "$STATE_FILE"
    echo "Done. All resources deleted."
}

# ============================================================================
# Main
# ============================================================================
case "${1:-help}" in
    check)        check ;;
    availability) availability ;;
    train)        train ;;
    wait)         wait_ready ;;
    upload)       upload ;;
    run)          run ;;
    download)     download ;;
    delete)       delete ;;
    status)       status ;;
    cleanup)      cleanup ;;
    help|*)
        cat << 'HELP'
Parameter Golf — RunPod CLI Workflow

  BEFORE TRAINING:
    check          Verify prerequisites, volume, GPU stock
    availability   H100 SXM stock by datacenter
    status         Show pods, volumes, SSH info

  TRAINING RUN (~$5.38 on-demand, ~$3 spot):
    train          Create 8x H100 pod with network volume
    wait           Wait for pod ready, show SSH info
    upload         SCP submission script to pod
    run            Execute training (~10 min)
    download       SCP results back
    delete         DELETE pod (stop billing!)

  END OF COMPETITION:
    cleanup        Delete everything (pods + volume)

  TYPICAL FLOW:
    ./runpod_cli.sh check
    ./runpod_cli.sh train
    ./runpod_cli.sh wait
    ./runpod_cli.sh upload
    ./runpod_cli.sh run
    ./runpod_cli.sh download
    ./runpod_cli.sh delete

  Volume: qwervlzocy (US-GA-2, 50GB, has 80 train shards)
  Data persists across pod deletions.
HELP
        ;;
esac
