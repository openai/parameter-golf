#!/usr/bin/env bash
# Pull cubric lite training logs from RunPod for PR submission
set -euo pipefail

SSH_TARGET="${1:?Usage: $0 <ssh_user_host>}"
SSH_KEY="$HOME/.ssh/id_ed25519_apollo"
REMOTE_DIR="/workspace/parameter-golf"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/records/track_10min_16mb/2026-03-25_PodracerIII_cubric_lite_8xH100"

mkdir -p "$LOCAL_DIR"

echo "==> Listing remote log files..."
REMOTE_LOGS=$(echo "ls -1 ${REMOTE_DIR}/logs/podracer_red_*.log; exit" \
    | ssh -tt -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
          -i "$SSH_KEY" "$SSH_TARGET" 2>/dev/null \
    | tr -d '\r' \
    | sed 's/\x1b\[[?0-9;]*[a-zA-Z]//g' \
    | sed 's/\x1b\][^\x07]*\x07//g' \
    | grep podracer_red || true)

echo "$REMOTE_LOGS"
echo ""

pull_log() {
    local remote_path="$1"
    local local_name="$2"
    local local_path="${LOCAL_DIR}/${local_name}"
    local MARKER_START="===XFER_START_$(date +%s)==="
    local MARKER_END="===XFER_END_$(date +%s)==="

    echo "==> Pulling $(basename "$remote_path") -> $local_name"

    echo "echo '${MARKER_START}'; base64 '${remote_path}'; echo '${MARKER_END}'; exit" \
        | ssh -tt -o ConnectTimeout=15 -i "$SSH_KEY" "$SSH_TARGET" 2>/dev/null \
        | tr -d '\r' \
        | sed 's/\x1b\[[?0-9;]*[a-zA-Z]//g' \
        | sed 's/\x1b\][^\x07]*\x07//g' \
        > "/tmp/_pull_raw_$$.txt"

    sed -n "/^${MARKER_START}/,/^${MARKER_END}/{ /${MARKER_START}/d; /${MARKER_END}/d; p; }" \
        "/tmp/_pull_raw_$$.txt" \
        | base64 -d > "$local_path"

    local LOCAL_SIZE=$(wc -c < "$local_path")
    echo "    OK: $local_path ($LOCAL_SIZE bytes)"
    rm -f "/tmp/_pull_raw_$$.txt"
}

# Pull each log — we need to find the right files by seed
# List all podracer_red logs and pull them
for remote_log in $REMOTE_LOGS; do
    remote_log=$(echo "$remote_log" | tr -d '[:space:]')
    [ -z "$remote_log" ] && continue
    local_name=$(basename "$remote_log")
    # Rename to match submission convention based on seed in filename
    if echo "$local_name" | grep -q "s2045"; then
        pull_log "$remote_log" "train_seed2045.log"
    elif echo "$local_name" | grep -q "s43"; then
        pull_log "$remote_log" "train_seed43.log"
    elif echo "$local_name" | grep -q "s7_"; then
        pull_log "$remote_log" "train_seed7.log"
    elif echo "$local_name" | grep -q "s42"; then
        pull_log "$remote_log" "train_seed42.log"
    else
        pull_log "$remote_log" "$local_name"
    fi
done

# Also pull the model checkpoint
echo ""
echo "==> Pulling model checkpoint..."
pull_log "${REMOTE_DIR}/final_model.int6.ptz" "final_model.int6.ptz"

echo ""
echo "==> Done. Files in: $LOCAL_DIR"
ls -lh "$LOCAL_DIR"/
