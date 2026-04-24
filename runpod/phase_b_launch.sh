#!/bin/bash
# Phase B launcher — run from local Mac after Phase A HF upload is done.
# Creates an 8xH100 Secure pod in AP-IN-1 (the only DC with 8xH100 Secure),
# waits for SSH, then runs bootstrap + train baseline + sweep.
#
# Requires env:
#   - runpodctl configured with API key (~/.runpod/config.toml already set)
#   - HF_TOKEN available for the pod (passed via --env)
#
# Cost guard: defaults to 60 min max pod time via MAX_POD_MINUTES — change at top.

set -euo pipefail

MAX_POD_MINUTES=${MAX_POD_MINUTES:-90}
POD_NAME="pg-ngram-phb-$(date +%Y%m%d-%H%M)"
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.config/parameter-golf/hf_token}
HF_TOKEN_VAL=$(cat "$HF_TOKEN_FILE")

echo "[launch] Creating 8xH100 Secure AP-IN-1 pod: $POD_NAME"
CREATE_OUT=$(runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "NVIDIA H100 80GB HBM3" \
    --gpuCount 8 \
    --secureCloud \
    --imageName "runpod/parameter-golf:latest" \
    --containerDiskSize 80 \
    --startSSH \
    --ports "22/tcp" \
    --env "HF_TOKEN=$HF_TOKEN_VAL" \
    --mem 64 --vcpu 32 \
    2>&1 | head -3)
echo "[launch] $CREATE_OUT"
POD_ID=$(echo "$CREATE_OUT" | grep -oE 'pod "[a-z0-9]+"' | sed 's/pod "\(.*\)"/\1/' | head -1)
if [ -z "$POD_ID" ]; then
  echo "[launch] FAILED to create pod — aborting"
  exit 1
fi
echo "[launch] POD_ID=$POD_ID"
echo "POD_ID=$POD_ID" > /tmp/pg_phase_b_pod.env

# Kill-switch: auto-stop after $MAX_POD_MINUTES
( sleep $((MAX_POD_MINUTES * 60)) && echo "[launch] AUTO-STOP after $MAX_POD_MINUTES min" && runpodctl stop pod "$POD_ID" ) &
KILL_PID=$!
echo "[launch] auto-stop scheduled (PID $KILL_PID)"

# Poll for SSH
echo "[launch] waiting for pod to get a public IP..."
SSH_IP=""; SSH_PORT=""
for i in $(seq 1 60); do
  POD_JSON=$(curl -sS -X POST "https://api.runpod.io/graphql?api_key=$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query { pod(input:{podId:\\\"$POD_ID\\\"}) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }\"}")
  SSH_IP=$(echo "$POD_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); ports=(d.get('data',{}).get('pod',{}) or {}).get('runtime',{}).get('ports',[]) or []; p=[x for x in ports if x.get('privatePort')==22 and x.get('isIpPublic')]; print(p[0]['ip'] if p else '')" 2>/dev/null || echo "")
  SSH_PORT=$(echo "$POD_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); ports=(d.get('data',{}).get('pod',{}) or {}).get('runtime',{}).get('ports',[]) or []; p=[x for x in ports if x.get('privatePort')==22 and x.get('isIpPublic')]; print(p[0]['publicPort'] if p else '')" 2>/dev/null || echo "")
  if [ -n "$SSH_IP" ] && [ -n "$SSH_PORT" ]; then break; fi
  sleep 5
done
if [ -z "$SSH_IP" ]; then
  echo "[launch] pod failed to get SSH IP after 5 min"
  kill $KILL_PID 2>/dev/null || true
  runpodctl stop pod "$POD_ID"
  exit 1
fi
echo "[launch] pod SSH ready: $SSH_IP:$SSH_PORT"
echo "export SSH_IP=$SSH_IP SSH_PORT=$SSH_PORT POD_ID=$POD_ID" > /tmp/pg_phase_b_pod.env

# Wait for sshd to accept
echo "[launch] waiting for sshd..."
for i in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -i ~/.ssh/id_runpod -p $SSH_PORT root@$SSH_IP 'echo up' 2>/dev/null; then
    break
  fi
  sleep 5
done

echo "[launch] pod ready for commands. Next: run phase_b_onpod.sh on the pod."
echo "  ssh -i ~/.ssh/id_runpod -p $SSH_PORT root@$SSH_IP"
