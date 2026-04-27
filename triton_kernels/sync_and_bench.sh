#!/bin/bash
# Copy triton_kernels/ files to the pod and run bench_mamba3_bwd.py there.
#
# Usage:
#   bash triton_kernels/sync_and_bench.sh                                # baseline
#   bash triton_kernels/sync_and_bench.sh --save triton_kernels/ref.pt   # capture ref grads
#   MAMBA3_FUSED_BWD=1 bash triton_kernels/sync_and_bench.sh \
#       --check triton_kernels/ref.pt                                    # test fused kernel
#
# Forwards MAMBA3_FUSED_BWD into the remote shell. All trailing args are
# passed to the bench script.

set -euo pipefail

POD_HOST="${POD_HOST:-root@64.247.201.36}"
POD_PORT="${POD_PORT:-19614}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/parameter-golf}"

SCP_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$POD_PORT")
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$POD_PORT")

cd "$(dirname "$0")/.."

echo "==> Copying triton_kernels/ files to ${POD_HOST}:${REMOTE_DIR}/triton_kernels/"
scp "${SCP_OPTS[@]}" \
    triton_kernels/bench_mamba3_bwd.py \
    triton_kernels/mamba3_siso_bwd.py \
    triton_kernels/mamba3_siso_combined.py \
    "${POD_HOST}:${REMOTE_DIR}/triton_kernels/"

FUSED_FLAG="${MAMBA3_FUSED_BWD:-0}"
echo "==> Running bench (MAMBA3_FUSED_BWD=${FUSED_FLAG})..."
ssh "${SSH_OPTS[@]}" "${POD_HOST}" \
    "cd ${REMOTE_DIR} && MAMBA3_FUSED_BWD=${FUSED_FLAG} python3 triton_kernels/bench_mamba3_bwd.py $*"
