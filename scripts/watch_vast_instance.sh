#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519_apollo}"
LOG_FILE="${LOG_FILE:-logs/watch_vast_instance_${INSTANCE_ID}.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

check_ssh() {
  local host="$1"
  local port="$2"

  if [[ -z "${host}" || -z "${port}" ]]; then
    echo "skip"
    return 0
  fi

  if timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no \
    -i "${SSH_KEY_PATH}" -p "${port}" "root@${host}" "nvidia-smi -L | head -1" >/dev/null 2>&1; then
    echo "ok"
  else
    echo "fail"
  fi
}

while true; do
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  raw="$(vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null || true)"

  if [[ -z "${raw}" ]]; then
    echo "${ts} instance=${INSTANCE_ID} status=unknown error=vastai_show_failed" | tee -a "${LOG_FILE}"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  status="$(echo "${raw}" | jq -r '.actual_status // .cur_state // "unknown"' 2>/dev/null || echo "unknown")"
  intended="$(echo "${raw}" | jq -r '.intended_status // "unknown"' 2>/dev/null || echo "unknown")"
  gpu_name="$(echo "${raw}" | jq -r '.gpu_name // "unknown"' 2>/dev/null || echo "unknown")"
  num_gpus="$(echo "${raw}" | jq -r '.num_gpus // "?"' 2>/dev/null || echo "?")"
  host="$(echo "${raw}" | jq -r '.public_ipaddr // empty' 2>/dev/null || true)"
  port="$(echo "${raw}" | jq -r '.direct_port_start // (.ports["22/tcp"][0].HostPort // empty)' 2>/dev/null || true)"
  dph_total="$(echo "${raw}" | jq -r '.dph_total // "?"' 2>/dev/null || echo "?")"
  time_remaining="$(echo "${raw}" | jq -r '.time_remaining // "?"' 2>/dev/null || echo "?")"

  ssh_health="skip"
  if [[ "${status}" == "running" ]]; then
    ssh_health="$(check_ssh "${host}" "${port}")"
  fi

  echo "${ts} instance=${INSTANCE_ID} status=${status} intended=${intended} gpus=${num_gpus} gpu='${gpu_name}' host=${host:-na} port=${port:-na} ssh=${ssh_health} dph_total=${dph_total} remaining='${time_remaining}'" | tee -a "${LOG_FILE}"
  sleep "${INTERVAL_SECONDS}"
done
