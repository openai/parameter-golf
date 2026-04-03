#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <host> <port> [user]" >&2
  echo "example: $0 103.207.149.123 19217 root" >&2
  exit 2
fi

HOST="$1"
PORT="$2"
USER_NAME="${3:-root}"

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RECORD_DIR}/../../.." && pwd)"
SSH_KEY="${SFW_POD_SSH_KEY:-$HOME/.ssh/id_runpod}"
REMOTE_PARENT="${SFW_REMOTE_PARENT:-/workspace}"
REMOTE_REPO="${SFW_REMOTE_REPO:-${REMOTE_PARENT}/parameter-golf}"
TRAIN_SHARDS="${SFW_BOOTSTRAP_TRAIN_SHARDS:-80}"
DRY_RUN="${SFW_DRY_RUN:-0}"
REMOTE_BASE_REF="${SFW_REMOTE_BASE_REF:-main}"

SSH_CMD=(ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" -p "${PORT}" "${USER_NAME}@${HOST}")

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

echo "[prepare] host=${HOST} port=${PORT} user=${USER_NAME}"
echo "[prepare] ssh_key=${SSH_KEY}"
echo "[prepare] remote_repo=${REMOTE_REPO}"
echo "[prepare] train_shards=${TRAIN_SHARDS}"
echo "[prepare] remote_base_ref=${REMOTE_BASE_REF}"

run_cmd "${SSH_CMD[@]}" "
  set -euo pipefail
  mkdir -p '${REMOTE_PARENT}'
  cd '${REMOTE_PARENT}'
  if [[ -d parameter-golf && ! -d parameter-golf/.git ]]; then
    rm -rf parameter-golf
  fi
  if [[ -d parameter-golf/.git ]]; then
    cd parameter-golf
    git fetch origin --prune
    git checkout -f '${REMOTE_BASE_REF}'
    git reset --hard 'origin/${REMOTE_BASE_REF}'
  else
    git clone https://github.com/openai/parameter-golf.git
    cd parameter-golf
    git checkout -f '${REMOTE_BASE_REF}'
  fi
"

RECORD_REL="records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233"
OVERLAY_FILES=(
  "spectral_flood_walk_v2a_residual.py"
  "spectral_flood_walk_v2a1_host1233.py"
  "tests/test_spectral_flood_walk_v2a_residual.py"
  "tools/summarize_v2a1_host1233_runs.py"
  "${RECORD_REL}/.gitignore"
  "${RECORD_REL}/README.md"
  "${RECORD_REL}/bootstrap_remote.sh"
  "${RECORD_REL}/prepare_fresh_pod.sh"
  "${RECORD_REL}/promote_run.sh"
  "${RECORD_REL}/runpod_common.sh"
  "${RECORD_REL}/runpod_full.sh"
  "${RECORD_REL}/runpod_preflight.sh"
  "${RECORD_REL}/runpod_smoke.sh"
  "${RECORD_REL}/runpod_three_seeds.sh"
  "${RECORD_REL}/runpod_8x_smoke.sh"
  "${RECORD_REL}/runpod_8x_signcheck.sh"
  "${RECORD_REL}/runpod_8x_fullval.sh"
  "${RECORD_REL}/train_gpt.py"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] tar overlay from ${REPO_ROOT} to ${REMOTE_REPO}"
else
  COPYFILE_DISABLE=1 tar czf - -C "${REPO_ROOT}" "${OVERLAY_FILES[@]}" \
    | "${SSH_CMD[@]}" "cd '${REMOTE_REPO}' && tar xzf -"
fi

run_cmd "${SSH_CMD[@]}" "
  set -euo pipefail
  chmod +x '${REMOTE_REPO}/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233/'*.sh
  cd '${REMOTE_REPO}/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233'
  SFW_BOOTSTRAP_TRAIN_SHARDS='${TRAIN_SHARDS}' ./bootstrap_remote.sh
"

echo
echo "[prepare] fresh pod should now be ready"
echo "[prepare] next commands:"
echo "  ${SSH_CMD[*]}"
echo "  cd ${REMOTE_REPO}/records/track_non_record_16mb/2026-03-31_SpectralFloodWalk_v2a1_host1233"
echo "  ./runpod_preflight.sh"
echo "  SFW_TARGET_GPU_COUNT=8 SFW_NPROC_PER_NODE=8 ./runpod_smoke.sh"
echo "  SFW_TARGET_GPU_COUNT=8 SFW_NPROC_PER_NODE=8 SFW_VAL_TOKEN_LIMIT=4194304 ./runpod_full.sh"
