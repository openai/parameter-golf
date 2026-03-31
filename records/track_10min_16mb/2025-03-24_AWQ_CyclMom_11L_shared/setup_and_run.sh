#!/bin/bash
# ============================================================
# FULL SETUP + RUN for 8×H100 submission
# Usage: Pass SSH host and port as arguments
#   ./setup_and_run.sh <host> <port>
# ============================================================
set -euo pipefail

HOST="${1:?Usage: $0 <host> <port>}"
PORT="${2:?Usage: $0 <host> <port>}"
SSH="ssh -o StrictHostKeyChecking=no -p ${PORT} root@${HOST}"
SCP="scp -P ${PORT}"
LOCAL_BASE="/Users/sidhantthole/Documents/llama_index_exp/openaigolf/parameter-golf"
REMOTE_BASE="/workspace/parameter-golf"
EXP="submission_exp18_awq-cyclic-relusq-11Lshared_8xH100"

echo "=== Step 1: Test connection ==="
$SSH "echo 'Connected!' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

echo "=== Step 2: Clone repo + install deps ==="
$SSH "cd /workspace && rm -rf parameter-golf && git clone https://github.com/openai/parameter-golf.git && pip install --break-system-packages -q zstandard"

echo "=== Step 3: Start data download in background ==="
$SSH "cd ${REMOTE_BASE} && nohup python3 data/cached_challenge_fineweb.py --variant sp1024 > /tmp/data_download.out 2>&1 &"

echo "=== Step 4: Copy experiment files while data downloads ==="
$SSH "mkdir -p ${REMOTE_BASE}/records/h100_experiments"
$SCP -r "${LOCAL_BASE}/records/h100_experiments/${EXP}" "root@${HOST}:${REMOTE_BASE}/records/h100_experiments/"
$SSH "chmod +x ${REMOTE_BASE}/records/h100_experiments/${EXP}/run.sh"

echo "=== Step 5: Wait for data download ==="
while true; do
    COUNT=$($SSH "ls ${REMOTE_BASE}/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    echo "  Data shards: ${COUNT}/80"
    if [ "$COUNT" -ge 80 ]; then
        break
    fi
    sleep 10
done
echo "Data download complete!"

echo "=== Step 6: Launch submission (3 seeds) ==="
$SSH "nohup bash ${REMOTE_BASE}/records/h100_experiments/${EXP}/run.sh > /tmp/submission.out 2>&1 &"
echo "Submission launched!"

echo "=== Step 7: Syncing records every 10 seconds ==="
while true; do
    rsync -az -e "ssh -p ${PORT}" "root@${HOST}:${REMOTE_BASE}/records/h100_experiments/${EXP}/" "${LOCAL_BASE}/records/h100_experiments/${EXP}/" 2>/dev/null
    rsync -az -e "ssh -p ${PORT}" "root@${HOST}:${REMOTE_BASE}/logs/" "${LOCAL_BASE}/logs/" 2>/dev/null

    # Check if still running
    RUNNING=$($SSH "ps aux | grep train_gpt | grep -v grep | wc -l" 2>/dev/null || echo "0")
    if [ "$RUNNING" -eq 0 ]; then
        # Final sync
        rsync -az -e "ssh -p ${PORT}" "root@${HOST}:${REMOTE_BASE}/records/h100_experiments/${EXP}/" "${LOCAL_BASE}/records/h100_experiments/${EXP}/" 2>/dev/null
        rsync -az -e "ssh -p ${PORT}" "root@${HOST}:${REMOTE_BASE}/logs/" "${LOCAL_BASE}/logs/" 2>/dev/null
        echo "=== ALL SEEDS COMPLETE ==="
        break
    fi
    sleep 10
done
