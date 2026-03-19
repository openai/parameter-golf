#!/bin/bash
# RunPod management for parameter-golf
# Usage:
#   ./pod.sh create [gpus]   — spin up H100 pod (default 1 GPU)
#   ./pod.sh ssh              — SSH into the running pod
#   ./pod.sh status           — show pod status
#   ./pod.sh sync             — rsync project files to pod
#   ./pod.sh run              — sync + start autoresearch on the pod
#   ./pod.sh stop             — stop the pod (keeps volume)
#   ./pod.sh destroy          — terminate the pod

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RUNPOD_API_KEY=$(grep RUNPOD_API_KEY "$SCRIPT_DIR/.env.local" | cut -d= -f2)

POD_ID_FILE="$SCRIPT_DIR/.pod_id"
TEMPLATE_ID="y5cejece4j"
GPU_ID="NVIDIA H100 80GB HBM3"
REMOTE_DIR="/workspace/parameter-golf"

get_pod_id() {
    if [ -f "$POD_ID_FILE" ]; then
        cat "$POD_ID_FILE"
    else
        echo ""
    fi
}

get_pod_info() {
    local pod_id=$(get_pod_id)
    if [ -z "$pod_id" ]; then
        echo "No pod ID saved. Run './pod.sh create' first." >&2
        exit 1
    fi
    runpodctl pod get "$pod_id" -o json 2>/dev/null
}

get_ssh_parts() {
    # Outputs "host port" on stdout
    local info=$(get_pod_info)
    echo "$info" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pod = data if isinstance(data, dict) else data[0]
runtime = pod.get('runtime', {})
if runtime:
    ports = runtime.get('ports', [])
    for p in ports:
        if p.get('privatePort') == 22:
            print(f\"{p.get('ip')} {p.get('publicPort')}\")
            break
" 2>/dev/null
}

case "${1:-status}" in
    create)
        GPU_COUNT="${2:-1}"
        echo "Creating pod with ${GPU_COUNT}x ${GPU_ID}..."

        RESULT=$(runpodctl pod create \
            --name "parameter-golf" \
            --gpu-id "$GPU_ID" \
            --gpu-count "$GPU_COUNT" \
            --template-id "$TEMPLATE_ID" \
            --container-disk-in-gb 50 \
            --volume-in-gb 50 \
            --cloud-type SECURE \
            --ssh \
            -o json 2>&1)

        echo "$RESULT"

        POD_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || echo "")
        if [ -n "$POD_ID" ] && [ "$POD_ID" != "" ]; then
            echo "$POD_ID" > "$POD_ID_FILE"
            echo ""
            echo "Pod ID saved: $POD_ID"
            echo "Run './pod.sh status' to check, then './pod.sh ssh' to connect."
        else
            echo "Could not extract pod ID from response."
        fi
        ;;

    status)
        pod_id=$(get_pod_id)
        if [ -z "$pod_id" ]; then
            echo "No pod. Run './pod.sh create' first."
            exit 0
        fi
        echo "Pod ID: $pod_id"
        info=$(get_pod_info)
        echo "$info" | python3 -c "
import sys, json
data = json.load(sys.stdin)
pod = data if isinstance(data, dict) else data[0]
print(f\"Status:  {pod.get('desiredStatus', '?')} / {pod.get('runtime', {}).get('uptimeInSeconds', 0):.0f}s uptime\")
print(f\"GPU:     {pod.get('machine', {}).get('gpuDisplayName', '?')} x{pod.get('gpuCount', '?')}\")
print(f\"Cost:    \${pod.get('costPerHr', 0):.2f}/hr\")
runtime = pod.get('runtime', {})
if runtime:
    ports = runtime.get('ports', [])
    for p in ports:
        if p.get('privatePort') == 22:
            print(f\"SSH:     ssh root@{p['ip']} -p {p['publicPort']}\")
" 2>/dev/null || echo "$info"
        ;;

    ssh)
        PARTS=$(get_ssh_parts)
        if [ -z "$PARTS" ]; then
            echo "Pod not ready or no SSH port found. Check './pod.sh status'"
            exit 1
        fi
        HOST=$(echo "$PARTS" | cut -d' ' -f1)
        PORT=$(echo "$PARTS" | cut -d' ' -f2)
        echo "Connecting: ssh root@$HOST -p $PORT"
        ssh root@"$HOST" -p "$PORT" -o StrictHostKeyChecking=no
        ;;

    sync)
        PARTS=$(get_ssh_parts)
        if [ -z "$PARTS" ]; then
            echo "Pod not ready. Check './pod.sh status'"
            exit 1
        fi
        HOST=$(echo "$PARTS" | cut -d' ' -f1)
        PORT=$(echo "$PARTS" | cut -d' ' -f2)

        echo "Syncing project to pod..."
        rsync -avz --progress \
            -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
            --exclude '.git' \
            --exclude 'data/datasets' \
            --exclude 'data/tokenizers' \
            --exclude '__pycache__' \
            --exclude '*.pyc' \
            --exclude 'logs' \
            --exclude 'autoresearch/logs' \
            --exclude 'final_model*' \
            "$SCRIPT_DIR/" "root@$HOST:$REMOTE_DIR/"
        echo "Done. Files synced to $REMOTE_DIR/"
        ;;

    run)
        # Sync then start autoresearch
        "$0" sync

        PARTS=$(get_ssh_parts)
        HOST=$(echo "$PARTS" | cut -d' ' -f1)
        PORT=$(echo "$PARTS" | cut -d' ' -f2)

        echo ""
        echo "Starting autoresearch on pod..."
        if [ -n "$ANTHROPIC_API_KEY" ]; then
            ANTHRO_KEY="$ANTHROPIC_API_KEY"
        elif grep -q ANTHROPIC_API_KEY "$SCRIPT_DIR/.env.local" 2>/dev/null; then
            ANTHRO_KEY=$(grep ANTHROPIC_API_KEY "$SCRIPT_DIR/.env.local" | cut -d= -f2)
        else
            echo "Set ANTHROPIC_API_KEY in .env.local or environment"
            exit 1
        fi

        ssh -p "$PORT" -o StrictHostKeyChecking=no "root@$HOST" \
            "cd $REMOTE_DIR && \
             export ANTHROPIC_API_KEY='$ANTHRO_KEY' && \
             export EXPERIMENT_SECONDS=${EXPERIMENT_SECONDS:-180} && \
             export GPUS=${GPUS:-1} && \
             nohup python3 autoresearch.py > autoresearch.out 2>&1 &
             echo 'autoresearch started in background (PID: '\$!')'
             echo 'To follow: ssh in and run: tail -f $REMOTE_DIR/autoresearch.out'"
        ;;

    pull)
        PARTS=$(get_ssh_parts)
        if [ -z "$PARTS" ]; then
            echo "Pod not ready. Check './pod.sh status'"
            exit 1
        fi
        HOST=$(echo "$PARTS" | cut -d' ' -f1)
        PORT=$(echo "$PARTS" | cut -d' ' -f2)

        echo "Pulling experiment results from pod..."
        rsync -avz --progress \
            -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
            "root@$HOST:$REMOTE_DIR/autoresearch/experiments/" \
            "$SCRIPT_DIR/autoresearch/experiments/"
        rsync -avz \
            -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
            "root@$HOST:$REMOTE_DIR/autoresearch/history.jsonl" \
            "$SCRIPT_DIR/autoresearch/history.jsonl"
        rsync -avz \
            -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
            "root@$HOST:$REMOTE_DIR/autoresearch/train_gpt.best.py" \
            "$SCRIPT_DIR/autoresearch/train_gpt.best.py" 2>/dev/null || true
        echo "Done. Results in autoresearch/experiments/"
        ;;

    stop)
        pod_id=$(get_pod_id)
        if [ -z "$pod_id" ]; then
            echo "No pod to stop."
            exit 0
        fi
        echo "Stopping pod $pod_id..."
        runpodctl pod stop "$pod_id"
        echo "Pod stopped (volume preserved)."
        ;;

    destroy)
        pod_id=$(get_pod_id)
        if [ -z "$pod_id" ]; then
            echo "No pod to destroy."
            exit 0
        fi
        echo "Terminating pod $pod_id..."
        runpodctl pod remove "$pod_id"
        rm -f "$POD_ID_FILE"
        echo "Pod terminated."
        ;;

    *)
        echo "Usage: ./pod.sh {create|status|ssh|sync|run|pull|stop|destroy}"
        echo ""
        echo "  create [gpus]  — spin up H100 pod (default 1 GPU)"
        echo "  status         — show pod info"
        echo "  ssh            — connect to pod"
        echo "  sync           — rsync project files to pod"
        echo "  run            — sync + start autoresearch in background"
        echo "  pull           — pull experiment results from pod"
        echo "  stop           — pause pod (keeps volume/data)"
        echo "  destroy        — terminate pod completely"
        ;;
esac
