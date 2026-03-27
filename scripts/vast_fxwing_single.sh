#!/usr/bin/env bash
# vast_fxwing_single.sh — Rent a single GPU on Vast.ai, run FX-Wing, pull results.
#
# Usage:
#   bash scripts/vast_fxwing_single.sh
#   bash scripts/vast_fxwing_single.sh --price 3.00 --gpu RTX_4090
#   bash scripts/vast_fxwing_single.sh --keep-instance   # don't destroy after run
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_API_KEY
#   SSH key at ~/.ssh/id_ed25519_apollo registered on vast.ai

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GPU="${GPU:-H100_SXM}"
NUM_GPUS=1
MIN_RELIABILITY="${MIN_RELIABILITY:-0.90}"
MAX_PRICE="${MAX_PRICE:-4.00}"
DISK_GB=60
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_apollo}"
IMAGE="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
SEED="${SEED:-1337}"
WALLCLOCK="${WALLCLOCK:-600}"
KEEP_INSTANCE="${KEEP_INSTANCE:-0}"
AUTO_YES="${AUTO_YES:-1}"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BRANCH="test"
REPO_URL="https://github.com/newjordan/parameter-golf.git"
RESULTS_DIR="${LOCAL_DIR}/results/fxwing_vast_$(date +%Y%m%d_%H%M%S)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="fxwing_1gpu_${TIMESTAMP}"

INSTANCE_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --price)         MAX_PRICE="$2";     shift 2 ;;
    --gpu)           GPU="$2";           shift 2 ;;
    --seed)          SEED="$2";          shift 2 ;;
    --wallclock)     WALLCLOCK="$2";     shift 2 ;;
    --keep-instance) KEEP_INSTANCE=1;    shift 1 ;;
    --no-auto-yes)   AUTO_YES=0;         shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cleanup() {
  set +e
  if [[ -n "${INSTANCE_ID}" && "${KEEP_INSTANCE}" != "1" ]]; then
    echo "==> Destroying instance ${INSTANCE_ID}..."
    vastai destroy instance "${INSTANCE_ID}" >/dev/null 2>&1 || true
    echo "==> Destroyed."
  elif [[ -n "${INSTANCE_ID}" ]]; then
    echo "==> KEEP_INSTANCE=1 — instance ${INSTANCE_ID} left running."
  fi
}
trap cleanup EXIT

echo "============================================"
echo "  Vast.ai FX-Wing Single GPU"
echo "  GPU: ${GPU}  Max: \$${MAX_PRICE}/hr"
echo "  Seed: ${SEED}  Wallclock: ${WALLCLOCK}s"
echo "  Label: ${RUN_LABEL}"
echo "============================================"

command -v vastai >/dev/null || { echo "ERROR: vastai CLI not installed. pip install vastai"; exit 1; }
[[ -f "${SSH_KEY}" ]] || { echo "ERROR: SSH key missing at ${SSH_KEY}"; exit 1; }

# ── Find offer ────────────────────────────────────────────────────────────────
echo "==> Searching for 1x${GPU} offers (on-demand, <= \$${MAX_PRICE}/hr)..."
OFFER_JSON="$(vastai search offers "gpu_name=${GPU} num_gpus=1 reliability>${MIN_RELIABILITY} rentable=True" -t on-demand -o dph_total --raw 2>/dev/null)"
[[ -n "${OFFER_JSON}" && "${OFFER_JSON}" != "[]" ]] || { echo "ERROR: No ${GPU} offers found"; exit 1; }

OFFER_ROW="$(echo "${OFFER_JSON}" | jq -c --arg max "${MAX_PRICE}" 'map(select((.dph_total // 1e9) <= ($max|tonumber))) | .[0]')"
[[ -n "${OFFER_ROW}" && "${OFFER_ROW}" != "null" ]] || { echo "ERROR: No offers at or below \$${MAX_PRICE}/hr for ${GPU}"; exit 1; }

OFFER_ID="$(echo "${OFFER_ROW}"   | jq -r '(.ask_contract_id // .id)')"
OFFER_PRICE="$(echo "${OFFER_ROW}" | jq -r '(.dph_total // 0) | tostring')"
OFFER_GPU="$(echo "${OFFER_ROW}"  | jq -r '(.gpu_name // "?")')"

echo "==> Selected: ID=${OFFER_ID}  GPU=${OFFER_GPU}  \$${OFFER_PRICE}/hr"

if [[ "${AUTO_YES}" != "1" ]]; then
  read -r -p "Rent this instance? [y/N] " ans
  [[ "${ans}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

# ── Create instance ───────────────────────────────────────────────────────────
echo "==> Creating instance..."
CREATE_OUT="$(vastai create instance "${OFFER_ID}" --image "${IMAGE}" --disk "${DISK_GB}" --ssh --direct --label "${RUN_LABEL}" 2>&1)"
echo "${CREATE_OUT}"
INSTANCE_ID="$(echo "${CREATE_OUT}" | grep -oE "new_contract['\"[:space:]]*:[[:space:]]*[0-9]+" | grep -oE '[0-9]+' | head -1)"
[[ -n "${INSTANCE_ID}" ]] || { echo "ERROR: could not parse instance id"; exit 1; }
echo "==> Instance ID: ${INSTANCE_ID}"

# ── Wait for running ──────────────────────────────────────────────────────────
WAITED=0; POLL=10; MAX_WAIT=600; STATUS="unknown"
echo "==> Waiting for running..."
while [[ ${WAITED} -lt ${MAX_WAIT} ]]; do
  STATUS="$(vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("actual_status","?"))' 2>/dev/null || echo unknown)"
  [[ "${STATUS}" == "running" ]] && break
  echo "    status=${STATUS} (${WAITED}s/${MAX_WAIT}s)"
  sleep ${POLL}; WAITED=$((WAITED + POLL))
done
[[ "${STATUS}" == "running" ]] || { echo "ERROR: instance never reached running state"; exit 1; }
sleep 5

# ── SSH details ───────────────────────────────────────────────────────────────
SSH_URL="$(vastai ssh-url "${INSTANCE_ID}" 2>/dev/null || true)"
if [[ "${SSH_URL}" == ssh://* ]]; then
  SSH_HOST="$(echo "${SSH_URL}" | sed -E 's#ssh://([^:]+):([0-9]+)#\1#')"
  SSH_PORT="$(echo "${SSH_URL}" | sed -E 's#ssh://([^:]+):([0-9]+)#\2#')"
else
  SSH_PORT="$(echo "${SSH_URL}" | grep -oE '\-p [0-9]+' | awk '{print $2}')"
  SSH_HOST="$(echo "${SSH_URL}" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | tail -1)"
fi
[[ -n "${SSH_PORT}" && -n "${SSH_HOST}" ]] || { echo "ERROR: invalid ssh url: ${SSH_URL}"; exit 1; }

SSH_CMD="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i ${SSH_KEY} -p ${SSH_PORT} ${SSH_HOST}"
SCP_CMD="scp -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i ${SSH_KEY} -P ${SSH_PORT}"

echo "==> Testing SSH (${SSH_HOST}:${SSH_PORT})..."
for i in 1 2 3 4 5 6 7 8; do
  if ${SSH_CMD} "echo OK" 2>/dev/null | grep -q OK; then break; fi
  sleep 5
  [[ $i -eq 8 ]] && { echo "ERROR: SSH not ready after 40s"; exit 1; }
done

# ── Setup repo ────────────────────────────────────────────────────────────────
echo "==> Cloning repo + installing deps..."
${SSH_CMD} "
  set -euo pipefail
  git clone -b ${BRANCH} ${REPO_URL} /workspace/parameter-golf-lab
  cd /workspace/parameter-golf-lab
  pip install -q sentencepiece zstandard || true
  mkdir -p data/datasets/fineweb10B_sp1024 data/tokenizers logs
  nvidia-smi -L || true
  python3 -V
"

# ── Upload data ───────────────────────────────────────────────────────────────
echo "==> Uploading data files..."
${SCP_CMD} \
  "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" \
  "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" \
  "${SSH_HOST}:/workspace/parameter-golf-lab/data/datasets/fineweb10B_sp1024/"
${SCP_CMD} \
  "${LOCAL_DIR}/data/tokenizers/fineweb_1024_bpe.model" \
  "${SSH_HOST}:/workspace/parameter-golf-lab/data/tokenizers/"
echo "==> Data uploaded."

# ── Run FX-Wing ───────────────────────────────────────────────────────────────
echo "==> Launching FX-Wing (NPROC=1, seed=${SEED}, wallclock=${WALLCLOCK}s)..."
mkdir -p "${RESULTS_DIR}"
RUN_LOG_LOCAL="${RESULTS_DIR}/fxwing_s${SEED}_${TIMESTAMP}.log"

${SSH_CMD} "
  set -euo pipefail
  cd /workspace/parameter-golf-lab
  SEED=${SEED} \
  NPROC_PER_NODE=1 \
  MAX_WALLCLOCK_SECONDS=${WALLCLOCK} \
  bash experiments/FX_Wing/run.sh
" 2>&1 | tee "${RUN_LOG_LOCAL}"

# ── Pull results ──────────────────────────────────────────────────────────────
echo "==> Pulling artifacts..."
${SCP_CMD} \
  "${SSH_HOST}:/workspace/parameter-golf-lab/final_model.pt" \
  "${SSH_HOST}:/workspace/parameter-golf-lab/final_model.int6.ptz" \
  "${RESULTS_DIR}/" 2>/dev/null || echo "  WARNING: some artifact files missing"
${SCP_CMD} \
  "${SSH_HOST}:/workspace/parameter-golf-lab/logs/fxwing_*.log" \
  "${RESULTS_DIR}/" 2>/dev/null || true

echo "============================================"
echo "  DONE — FX-Wing single GPU"
echo "  Results: ${RESULTS_DIR}/"
echo "  Log:     ${RUN_LOG_LOCAL}"
echo "============================================"
