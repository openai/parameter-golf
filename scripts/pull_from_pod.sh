#!/usr/bin/env bash
# pull_from_pod.sh — Pull training artifacts from a RunPod instance
#
# Usage:
#   ./scripts/pull_from_pod.sh <ssh_user_host> [label]

set -euo pipefail

SSH_TARGET="${1:?Usage: $0 <ssh_user_host> [label]}"
LABEL="${2:-$(date +%Y%m%d_%H%M%S)}"
SSH_KEY="$HOME/.ssh/id_ed25519_apollo"
REMOTE_DIR="/workspace/parameter-golf"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
MARKER_START="===XFER_START_$(date +%s)==="
MARKER_END="===XFER_END_$(date +%s)==="

mkdir -p "$LOCAL_DIR"

echo "==> Connecting to $SSH_TARGET"
echo "==> Label: $LABEL"
echo "==> Destination: $LOCAL_DIR"
echo ""

echo "==> Listing remote checkpoint files..."
REMOTE_FILES=$(echo "ls -lh ${REMOTE_DIR}/final_model*; exit" \
    | ssh -tt -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
          -i "$SSH_KEY" "$SSH_TARGET" 2>/dev/null \
    | tr -d '\r' \
    | sed 's/\x1b\[[?0-9;]*[a-zA-Z]//g' \
    | sed 's/\x1b\][^\x07]*\x07//g' \
    | grep final_model || true)

if [ -z "$REMOTE_FILES" ]; then
    echo "ERROR: No final_model* files found in $REMOTE_DIR"
    exit 1
fi

echo "$REMOTE_FILES"
echo ""

FILES=$(echo "$REMOTE_FILES" | grep -oE 'final_model[^ ]+' | sort -u)

pull_file() {
    local remote_path="$1"
    local filename=$(basename "$remote_path")
    local local_path="${LOCAL_DIR}/${LABEL}_${filename}"

    echo "==> Pulling $filename..."

    echo "echo '${MARKER_START}'; base64 '${remote_path}'; echo '${MARKER_END}'; exit" \
        | ssh -tt -o ConnectTimeout=15 -i "$SSH_KEY" "$SSH_TARGET" 2>/dev/null \
        | tr -d '\r' \
        | sed 's/\x1b\[[?0-9;]*[a-zA-Z]//g' \
        | sed 's/\x1b\][^\x07]*\x07//g' \
        > "/tmp/_pull_raw_$$_${filename}.txt"

    sed -n "/^${MARKER_START}/,/^${MARKER_END}/{ /${MARKER_START}/d; /${MARKER_END}/d; p; }" \
        "/tmp/_pull_raw_$$_${filename}.txt" \
        | base64 -d > "$local_path"

    REMOTE_MD5=$(echo "md5sum '${remote_path}'; exit" \
        | ssh -tt -o ConnectTimeout=15 -i "$SSH_KEY" "$SSH_TARGET" 2>/dev/null \
        | tr -d '\r' \
        | sed 's/\x1b\[[?0-9;]*[a-zA-Z]//g' \
        | sed 's/\x1b\][^\x07]*\x07//g' \
        | grep "$filename" | grep -oE '^[a-f0-9]{32}' | tail -1)

    LOCAL_MD5=$(md5sum "$local_path" | cut -d' ' -f1)

    if [ "$REMOTE_MD5" = "$LOCAL_MD5" ]; then
        local size=$(ls -lh "$local_path" | awk '{print $5}')
        echo "    OK: $local_path ($size) MD5=$LOCAL_MD5"
    else
        echo "    FAIL: MD5 mismatch! remote=$REMOTE_MD5 local=$LOCAL_MD5"
        echo "    File saved but may be corrupt: $local_path"
        return 1
    fi

    rm -f "/tmp/_pull_raw_$$_${filename}.txt"
}

FAIL=0
for f in $FILES; do
    pull_file "${REMOTE_DIR}/${f}" || FAIL=1
done

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "==> All files pulled and verified!"
else
    echo "==> WARNING: Some files failed verification"
fi

echo "==> Checkpoints saved to: $LOCAL_DIR"
ls -lh "$LOCAL_DIR"/${LABEL}_* 2>/dev/null
