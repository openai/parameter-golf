#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VAST_BIN="${VAST_BIN:-$ROOT_DIR/.venv-vastai/bin/vastai}"
TEMPLATE_NAME="${PG_TEMPLATE_NAME:-parameter-golf-8xh100}"
IMAGE="${PG_TEMPLATE_IMAGE:-nvcr.io/nvidia/pytorch:25.12-py3}"
DISK_SPACE="${PG_TEMPLATE_DISK_GB:-300}"
DESC="${PG_TEMPLATE_DESC:-Parameter Golf 8xH100 SSH template with direct mode and shared memory configured.}"
SEARCH_PARAMS="${PG_TEMPLATE_SEARCH_PARAMS:-gpu_name in [H100_SXM,H100_NVL] num_gpus=8 reliability>=0.98 static_ip=True direct_port_count>=2 cpu_arch=amd64 disk_space>=300 rented=False}"
ENV_FLAGS="${PG_TEMPLATE_ENV:---shm-size=64g -p 22:22}"

cd "$ROOT_DIR"
"$VAST_BIN" create template \
    --name "$TEMPLATE_NAME" \
    --image "$IMAGE" \
    --env "$ENV_FLAGS" \
    --search_params "$SEARCH_PARAMS" \
    --disk_space "$DISK_SPACE" \
    --desc "$DESC" \
    --ssh \
    --direct
