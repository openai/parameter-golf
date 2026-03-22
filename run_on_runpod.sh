#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Parameter Golf — RunPod Deployment Script
# Usage:
#   ./run_on_runpod.sh              # Create spot pod, setup, train
#   ./run_on_runpod.sh --status     # Pod status + SSH command
#   ./run_on_runpod.sh --logs       # Tail training logs
#   ./run_on_runpod.sh --results    # Show key metrics
#   ./run_on_runpod.sh --save-log <tag>  # Save full log to logs/<timestamp>_<tag>.log
#   ./run_on_runpod.sh --stop       # Stop pod
#   ./run_on_runpod.sh --delete     # Delete pod
# =============================================================================

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PUB=$(cat "${SSH_KEY}.pub")
TEMPLATE_ID="y5cejece4j"
GPU_ID="${GPU_ID:-NVIDIA H100 80GB HBM3}"
GPU_COUNT="${GPU_COUNT:-1}"
BID_PRICE="${BID_PRICE:-1.75}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
POD_NAME="param-golf-run"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.runpod_state"
mkdir -p "$STATE_DIR"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"

# --- GraphQL helper ---
gql() {
  curl -s -X POST https://api.runpod.io/graphql \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "{\"query\": \"$1\"}"
}

# --- Get pod SSH info ---
get_pod_ssh() {
  RUNPOD_API_KEY=$RUNPOD_API_KEY runpodctl pod get "$1" 2>/dev/null | python3 -c "
import json,sys
d = json.load(sys.stdin)
ssh = d.get('ssh', {})
ip, port = ssh.get('ip',''), ssh.get('port','')
print(f'{ip} {port}') if ip and port else print('')"
}

# --- SSH to saved pod ---
pod_ssh() {
  local pod_id=$(cat "$STATE_DIR/pod_id" 2>/dev/null || true)
  [ -z "$pod_id" ] && { echo "No pod ID saved."; exit 1; }
  local ssh_info=$(get_pod_ssh "$pod_id")
  [ -z "$ssh_info" ] && { echo "Pod $pod_id not ready."; exit 1; }
  local ip=$(echo "$ssh_info" | cut -d' ' -f1)
  local port=$(echo "$ssh_info" | cut -d' ' -f2)
  ssh $SSH_OPTS -i "$SSH_KEY" "root@$ip" -p "$port" "$@"
}

# --- Subcommands ---
case "${1:-run}" in
  --status)
    pod_id=$(cat "$STATE_DIR/pod_id" 2>/dev/null || true)
    [ -z "$pod_id" ] && { echo "No pod."; exit 0; }
    RUNPOD_API_KEY=$RUNPOD_API_KEY runpodctl pod get "$pod_id" 2>/dev/null | python3 -c "
import json,sys; d=json.load(sys.stdin)
print(f'Pod: {d[\"id\"]}  Status: {d.get(\"desiredStatus\",\"?\")}  Cost: \${d.get(\"costPerHr\",\"?\")}/hr')
ssh=d.get('ssh',{}); ip=ssh.get('ip',''); port=ssh.get('port','')
if ip: print(f'SSH: ssh -i $SSH_KEY root@{ip} -p {port}')"
    exit 0 ;;
  --logs)
    pod_ssh "tail -30 /workspace/train_run.log 2>/dev/null || echo 'No logs'" ; exit 0 ;;
  --results)
    pod_ssh "grep -E 'val_bpb|val_loss|Serial|Total|stop|peak|swa:|late_qat|final_int|ttt:|model_params' /workspace/train_run.log 2>/dev/null" ; exit 0 ;;
  --stop)
    pod_id=$(cat "$STATE_DIR/pod_id" 2>/dev/null || true)
    [ -z "$pod_id" ] && { echo "No pod."; exit 0; }
    RUNPOD_API_KEY=$RUNPOD_API_KEY runpodctl pod stop "$pod_id" ; exit 0 ;;
  --delete)
    pod_id=$(cat "$STATE_DIR/pod_id" 2>/dev/null || true)
    [ -z "$pod_id" ] && { echo "No pod."; exit 0; }
    RUNPOD_API_KEY=$RUNPOD_API_KEY runpodctl pod delete "$pod_id"
    rm -f "$STATE_DIR/pod_id" ; exit 0 ;;
  --save-log)
    LOGS_DIR="$SCRIPT_DIR/logs"; mkdir -p "$LOGS_DIR"
    TS=$(date +%Y%m%d_%H%M%S); TAG="${2:-run}"
    pod_ssh "cat /workspace/train_run.log" > "$LOGS_DIR/${TS}_${TAG}.log"
    grep -E 'val_bpb|Serial|Total|stop|peak|swa:|late_qat|final_int|ttt:|model_params' "$LOGS_DIR/${TS}_${TAG}.log" > "$LOGS_DIR/${TS}_${TAG}.summary"
    echo "Saved: $LOGS_DIR/${TS}_${TAG}.log"
    cat "$LOGS_DIR/${TS}_${TAG}.summary" ; exit 0 ;;
  run|"") ;;
  *) echo "Unknown: $1"; exit 1 ;;
esac

# =============================================================================
# MAIN: Create pod → setup → train (optimized for speed)
# =============================================================================

echo "=== Parameter Golf RunPod Deploy ==="
echo "Creating spot $GPU_ID x$GPU_COUNT (\$$BID_PRICE/gpu/hr)..."

POD_RESULT=$(gql "mutation { podRentInterruptable(input: { name: \\\"$POD_NAME\\\", templateId: \\\"$TEMPLATE_ID\\\", gpuTypeId: \\\"$GPU_ID\\\", gpuCount: $GPU_COUNT, volumeInGb: 50, containerDiskInGb: 50, cloudType: SECURE, startSsh: true, ports: \\\"8888/http,22/tcp\\\", bidPerGpu: $BID_PRICE, env: [{key: \\\"JUPYTER_PASSWORD\\\", value: \\\"parameter-golf\\\"}, {key: \\\"PUBLIC_KEY\\\", value: \\\"$SSH_PUB\\\"}] }) { id costPerHr desiredStatus machine { gpuDisplayName location } } }")

POD_ID=$(echo "$POD_RESULT" | python3 -c "
import json,sys; d=json.load(sys.stdin)
pod=d.get('data',{}).get('podRentInterruptable')
if not pod: errs=d.get('errors',[]); print(f'ERROR: {errs[0][\"message\"] if errs else \"Unknown\"}',file=sys.stderr); sys.exit(1)
print(pod['id'])")
COST=$(echo "$POD_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin)['data']['podRentInterruptable']; print(f\"\${d['costPerHr']}/hr {d['machine']['gpuDisplayName']} ({d['machine']['location']})\")")
echo "$POD_ID" > "$STATE_DIR/pod_id"
echo "Pod: $POD_ID — $COST"

# Wait for SSH
echo -n "Waiting for SSH..."
IP=""; PORT=""
for i in $(seq 1 30); do
  sleep 10
  SSH_INFO=$(get_pod_ssh "$POD_ID")
  if [ -n "$SSH_INFO" ]; then
    IP=$(echo "$SSH_INFO" | cut -d' ' -f1)
    PORT=$(echo "$SSH_INFO" | cut -d' ' -f2)
    if ssh $SSH_OPTS -i "$SSH_KEY" "root@$IP" -p "$PORT" "echo ok" >/dev/null 2>&1; then
      echo " ready!"; break
    fi
  fi
  echo -n "."
done
[ -z "$IP" ] && { echo " TIMEOUT"; exit 1; }
echo "SSH: ssh -i $SSH_KEY root@$IP -p $PORT"
S="ssh $SSH_OPTS -i $SSH_KEY root@$IP -p $PORT"

# === SETUP: clone + deps + data (with nohup to survive SSH drops) ===
echo "Setting up pod..."
$S bash -c 'cat > /workspace/setup.sh << "SETUPEOF"
#!/bin/bash
set -e
cd /workspace
[ -d parameter-golf/.git ] || git clone --depth 1 https://github.com/openai/parameter-golf.git
python3 -c "import zstandard" 2>/dev/null || pip install --break-system-packages -q zstandard
cd parameter-golf
[ -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ] || \
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards '"$TRAIN_SHARDS"'
echo SETUP_OK > /workspace/.setup_done
SETUPEOF
chmod +x /workspace/setup.sh
nohup /workspace/setup.sh > /workspace/setup.log 2>&1 &
echo "setup PID=$!"'

# Upload train_gpt.py while setup runs
echo "Uploading train_gpt.py..."
scp $SSH_OPTS -i "$SSH_KEY" -P "$PORT" "$SCRIPT_DIR/train_gpt.py" "root@$IP:/workspace/parameter-golf/train_gpt.py" 2>/dev/null || true

# Wait for setup to finish (poll every 10s)
echo -n "Waiting for data download..."
for i in $(seq 1 120); do
  if $S "[ -f /workspace/.setup_done ] && echo done" 2>/dev/null | grep -q done; then
    echo " done!"
    break
  fi
  sleep 10
  echo -n "."
done

# Re-upload train_gpt.py (in case clone overwrote it)
scp $SSH_OPTS -i "$SSH_KEY" -P "$PORT" "$SCRIPT_DIR/train_gpt.py" "root@$IP:/workspace/parameter-golf/train_gpt.py"

# === LAUNCH TRAINING ===
echo "Starting training (nproc=$GPU_COUNT)..."
$S bash -c "cd /workspace/parameter-golf && nohup torchrun --standalone --nproc_per_node=$GPU_COUNT train_gpt.py > /workspace/train_run.log 2>&1 & echo PID=\$!"

echo ""
echo "=== Training started! ==="
echo "Monitor:  ./run_on_runpod.sh --logs"
echo "Results:  ./run_on_runpod.sh --results"
echo "Status:   ./run_on_runpod.sh --status"
echo "Save:     ./run_on_runpod.sh --save-log <tag>"
echo "Stop:     ./run_on_runpod.sh --stop"
echo "Delete:   ./run_on_runpod.sh --delete"
