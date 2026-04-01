#!/usr/bin/env bash
# Rent H100s on Vast.ai and run Cobra A/B.
# Supports 1x and 8x setups; defaults are tuned for stability.

set -euo pipefail

GPU="${GPU:-H100_SXM}"
NUM_GPUS="${NUM_GPUS:-1}"
NPROC="${NPROC:-${NUM_GPUS}}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.95}"
REQUIRE_VERIFIED="${REQUIRE_VERIFIED:-0}"
MAX_PRICE="${MAX_PRICE:-24.00}"
DISK_GB="${DISK_GB:-60}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_apollo}"

A_CAND="${A_CAND:-c0_green1_anchor}"
B_CAND="${B_CAND:-c1_complement_035}"
SEQUENCE="${SEQUENCE:-AB}"
SEEDS="${SEEDS:-1337}"
WALLCLOCK="${WALLCLOCK:-120}"
AUTO_YES="${AUTO_YES:-1}"
KEEP_INSTANCE="${KEEP_INSTANCE:-0}"

PROFILE_COMPILE_ENABLED="${PROFILE_COMPILE_ENABLED:-0}"
PROFILE_TORCHDYNAMO_DISABLE="${PROFILE_TORCHDYNAMO_DISABLE:-1}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-0}"
PROFILE_VAL_LOSS_EVERY="${PROFILE_VAL_LOSS_EVERY:-0}"
PROFILE_TRAIN_LOG_EVERY="${PROFILE_TRAIN_LOG_EVERY:-500}"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${LOCAL_DIR}/results/vast_cobra_ab"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL=""

INSTANCE_ID=""
PAYLOAD_DIR=""
TARBALL=""
SSH_CMD=""
SCP_CMD=""
SSH_HOST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --price) MAX_PRICE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --gpus) NUM_GPUS="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --a) A_CAND="$2"; shift 2 ;;
    --b) B_CAND="$2"; shift 2 ;;
    --sequence) SEQUENCE="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --wallclock) WALLCLOCK="$2"; shift 2 ;;
    --no-auto-yes) AUTO_YES=0; shift 1 ;;
    --keep-instance) KEEP_INSTANCE=1; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RUN_LABEL="cobra_ab_${NUM_GPUS}gpu_${TIMESTAMP}"

cleanup() {
  set +e
  if [[ -n "${INSTANCE_ID}" && "${KEEP_INSTANCE}" != "1" ]]; then
    echo "==> Destroying instance ${INSTANCE_ID}..."
    vastai destroy instance "${INSTANCE_ID}" >/dev/null 2>&1 || true
    echo "==> Destroyed."
  fi
  [[ -n "${PAYLOAD_DIR}" ]] && rm -rf "${PAYLOAD_DIR}" >/dev/null 2>&1 || true
  [[ -n "${TARBALL}" ]] && rm -f "${TARBALL}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "============================================"
echo "  Vast.ai Cobra A/B (${NUM_GPUS}x${GPU})"
echo "  Label: ${RUN_LABEL}"
echo "  Max price: \$${MAX_PRICE}/hr"
echo "  A: ${A_CAND}"
echo "  B: ${B_CAND}"
echo "  Sequence: ${SEQUENCE}"
echo "  Seeds: ${SEEDS}"
echo "  NPROC: ${NPROC}"
echo "  Wallclock per arm: ${WALLCLOCK}s"
echo "============================================"

command -v vastai >/dev/null || { echo "ERROR: vastai CLI not installed"; exit 1; }
[[ -f "${SSH_KEY}" ]] || { echo "ERROR: SSH key missing at ${SSH_KEY}"; exit 1; }

for f in \
  "${LOCAL_DIR}/experiments/Cobra/run_ab_sequence.py" \
  "${LOCAL_DIR}/experiments/Cobra/cobra_harness.py" \
  "${LOCAL_DIR}/experiments/Cobra/candidates.json" \
  "${LOCAL_DIR}/experiments/Cobra/profiles/cobra_base_quality.env" \
  "${LOCAL_DIR}/experiments/A_wing/green_1/train_gpt.py" \
  "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" \
  "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" \
  "${LOCAL_DIR}/data/tokenizers/fineweb_1024_bpe.model"
do
  [[ -f "${f}" ]] || { echo "ERROR: Missing ${f}"; exit 1; }
done

echo "==> Searching for ${NUM_GPUS}x${GPU} offers (price cap applied locally: <= \$${MAX_PRICE}/hr) ..."
OFFER_FILTER="gpu_name=${GPU} num_gpus=${NUM_GPUS} reliability>${MIN_RELIABILITY} rentable=True"
if [[ "${REQUIRE_VERIFIED}" == "1" ]]; then
  OFFER_FILTER="${OFFER_FILTER} verified=True"
fi
OFFER_JSON="$(vastai search offers "${OFFER_FILTER}" -t on-demand -o dph_total --raw 2>/dev/null)"
[[ -n "${OFFER_JSON}" && "${OFFER_JSON}" != "[]" ]] || { echo "ERROR: No matching offers from Vast"; exit 1; }

OFFER_ROW="$(echo "${OFFER_JSON}" | jq -c --arg max "${MAX_PRICE}" 'map(select((.dph_total // 1e9) <= ($max|tonumber))) | .[0]')"
[[ -n "${OFFER_ROW}" && "${OFFER_ROW}" != "null" ]] || { echo "ERROR: No offers at or below \$${MAX_PRICE}/hr"; exit 1; }

OFFER_ID="$(echo "${OFFER_ROW}" | jq -r '(.ask_contract_id // .id)')"
OFFER_PRICE="$(echo "${OFFER_ROW}" | jq -r '(.dph_total // 0) | tostring')"
OFFER_GPU="$(echo "${OFFER_ROW}" | jq -r '(.gpu_name // "?")')"

echo "==> Selected offer: ID=${OFFER_ID} ${OFFER_GPU} \$${OFFER_PRICE}/hr"

if [[ "${AUTO_YES}" != "1" ]]; then
  read -r -p "Rent this instance? [y/N] " ans
  [[ "${ans}" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

echo "==> Creating instance..."
CREATE_OUT="$(vastai create instance "${OFFER_ID}" --image "${IMAGE}" --disk "${DISK_GB}" --ssh --direct --label "${RUN_LABEL}" 2>&1)"
echo "${CREATE_OUT}"
INSTANCE_ID="$(echo "${CREATE_OUT}" | grep -oE "new_contract['\"[:space:]]*:[[:space:]]*[0-9]+" | grep -oE '[0-9]+' | head -1)"
[[ -n "${INSTANCE_ID}" ]] || { echo "ERROR: could not parse instance id"; exit 1; }
echo "==> Instance ID: ${INSTANCE_ID}"

WAITED=0
POLL=10
MAX_WAIT=600
STATUS="unknown"
echo "==> Waiting for running..."
while [[ ${WAITED} -lt ${MAX_WAIT} ]]; do
  STATUS="$(vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin).get("actual_status","?"))' 2>/dev/null || echo unknown)"
  [[ "${STATUS}" == "running" ]] && break
  echo "    status=${STATUS} (${WAITED}s/${MAX_WAIT}s)"
  sleep ${POLL}
  WAITED=$((WAITED + POLL))
done
[[ "${STATUS}" == "running" ]] || { echo "ERROR: instance not running"; exit 1; }

sleep 5
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
for i in 1 2 3 4 5 6; do
  if ${SSH_CMD} "echo OK" 2>/dev/null | grep -q OK; then
    break
  fi
  sleep 5
  [[ $i -eq 6 ]] && { echo "ERROR: SSH not ready"; exit 1; }
done

echo "==> Building payload..."
PAYLOAD_DIR="$(mktemp -d)"
mkdir -p "${PAYLOAD_DIR}/workspace/parameter-golf/experiments"
mkdir -p "${PAYLOAD_DIR}/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
mkdir -p "${PAYLOAD_DIR}/workspace/parameter-golf/data/tokenizers"
mkdir -p "${PAYLOAD_DIR}/workspace/parameter-golf/logs"

cp -r "${LOCAL_DIR}/experiments/Cobra" "${PAYLOAD_DIR}/workspace/parameter-golf/experiments/"
mkdir -p "${PAYLOAD_DIR}/workspace/parameter-golf/experiments/A_wing/green_1"
cp "${LOCAL_DIR}/experiments/A_wing/green_1/train_gpt.py" "${PAYLOAD_DIR}/workspace/parameter-golf/experiments/A_wing/green_1/"
cp "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin" "${PAYLOAD_DIR}/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/"
cp "${LOCAL_DIR}/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" "${PAYLOAD_DIR}/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/"
cp "${LOCAL_DIR}/data/tokenizers/fineweb_1024_bpe.model" "${PAYLOAD_DIR}/workspace/parameter-golf/data/tokenizers/"

TARBALL="/tmp/vast_cobra_ab_${TIMESTAMP}.tar.gz"
(cd "${PAYLOAD_DIR}/workspace/parameter-golf" && tar czf "${TARBALL}" .)
echo "==> Payload size: $(du -sh "${TARBALL}" | cut -f1)"

echo "==> Uploading payload..."
${SCP_CMD} "${TARBALL}" "${SSH_HOST}:/workspace/payload.tar.gz"

echo "==> Extracting payload + deps..."
${SSH_CMD} "
  set -euo pipefail
  mkdir -p /workspace/parameter-golf
  cd /workspace/parameter-golf
  tar xzf /workspace/payload.tar.gz
  pip install -q sentencepiece zstandard || true
  python3 -V
  nvidia-smi -L || true
"

echo "==> Running remote Cobra A/B..."
RUN_LOG_LOCAL="/tmp/vast_${RUN_LABEL}.log"
${SSH_CMD} "
  set -euo pipefail
  cd /workspace/parameter-golf
  PROFILE=experiments/Cobra/profiles/cobra_base_quality.env
  if grep -q '^COMPILE_ENABLED=' \$PROFILE; then sed -i 's/^COMPILE_ENABLED=.*/COMPILE_ENABLED=${PROFILE_COMPILE_ENABLED}/' \$PROFILE; else echo 'COMPILE_ENABLED=${PROFILE_COMPILE_ENABLED}' >> \$PROFILE; fi
  if grep -q '^TORCHDYNAMO_DISABLE=' \$PROFILE; then sed -i 's/^TORCHDYNAMO_DISABLE=.*/TORCHDYNAMO_DISABLE=${PROFILE_TORCHDYNAMO_DISABLE}/' \$PROFILE; else echo 'TORCHDYNAMO_DISABLE=${PROFILE_TORCHDYNAMO_DISABLE}' >> \$PROFILE; fi
  if grep -q '^WARMUP_STEPS=' \$PROFILE; then sed -i 's/^WARMUP_STEPS=.*/WARMUP_STEPS=${PROFILE_WARMUP_STEPS}/' \$PROFILE; else echo 'WARMUP_STEPS=${PROFILE_WARMUP_STEPS}' >> \$PROFILE; fi
  if grep -q '^VAL_LOSS_EVERY=' \$PROFILE; then sed -i 's/^VAL_LOSS_EVERY=.*/VAL_LOSS_EVERY=${PROFILE_VAL_LOSS_EVERY}/' \$PROFILE; else echo 'VAL_LOSS_EVERY=${PROFILE_VAL_LOSS_EVERY}' >> \$PROFILE; fi
  if grep -q '^TRAIN_LOG_EVERY=' \$PROFILE; then sed -i 's/^TRAIN_LOG_EVERY=.*/TRAIN_LOG_EVERY=${PROFILE_TRAIN_LOG_EVERY}/' \$PROFILE; else echo 'TRAIN_LOG_EVERY=${PROFILE_TRAIN_LOG_EVERY}' >> \$PROFILE; fi
  echo '--- remote profile overrides ---'
  grep -n -E '^(COMPILE_ENABLED|TORCHDYNAMO_DISABLE|WARMUP_STEPS|VAL_LOSS_EVERY|TRAIN_LOG_EVERY)=' \$PROFILE || true
  python3 experiments/Cobra/run_ab_sequence.py \
    --a ${A_CAND} \
    --b ${B_CAND} \
    --sequence ${SEQUENCE} \
    --seeds ${SEEDS} \
    --max-wallclock ${WALLCLOCK} \
    --nproc ${NPROC} \
    --execute
" | tee "${RUN_LOG_LOCAL}"

mkdir -p "${RESULTS_DIR}"
cp "${RUN_LOG_LOCAL}" "${RESULTS_DIR}/${RUN_LABEL}.log"
${SCP_CMD} "${SSH_HOST}:/workspace/parameter-golf/logs/cobra_*.log" "${RESULTS_DIR}/" 2>/dev/null || true

echo "============================================"
echo "DONE"
echo "Results log: ${RESULTS_DIR}/${RUN_LABEL}.log"
echo "Cobra logs : ${RESULTS_DIR}/cobra_*.log"
echo "============================================"
