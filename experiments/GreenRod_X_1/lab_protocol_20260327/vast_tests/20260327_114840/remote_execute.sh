#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="20260327_114840"
PAYLOAD="/workspace/lab_ab_payload_${RUN_TAG}.tar.gz"
REPO="/workspace/parameter-golf-lab"
TEST_DIR="${REPO}/experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/${RUN_TAG}"
RESULT_ROOT="${TEST_DIR}/remote_results"

mkdir -p /workspace
[[ -f "${PAYLOAD}" ]] || { echo "FATAL: missing payload ${PAYLOAD}"; exit 1; }

tar xzf "${PAYLOAD}" -C /
cd "${REPO}"

python3 -m pip install -q sentencepiece zstandard
python3 -m pip install -q fla-core==0.4.2 flash-linear-attention==0.4.2 || true

if python3 -c "from fla.layers.delta_net import DeltaNet" >/dev/null 2>&1; then
  ARMS_PATH="${TEST_DIR}/concept_arms.tsv"
  echo "FLA_OK=1 using ${ARMS_PATH}"
else
  ARMS_PATH="${TEST_DIR}/concept_arms_fallback.tsv"
  cat > "${ARMS_PATH}" <<'TSV'
arm	enabled	gdn_enabled	gdn_num_layers	gdn_lr	xsa_last_n	notes
control	1	0	0	0.0018	11	Fallback control only (fla unavailable)
TSV
  echo "FLA_OK=0 fallback arms=${ARMS_PATH}"
fi

mkdir -p "${RESULT_ROOT}"
export ARMS_TSV="${ARMS_PATH}"
export RESULT_DIR="${RESULT_ROOT}/ab1gpu_${RUN_TAG}"
export SEEDS="1337,1338"
export NPROC_PER_NODE=1
export MAX_WALLCLOCK_SECONDS=180
export VAL_LOSS_EVERY=200
export SKIP_FINAL_EVAL=1
export PROMOTE_DELTA=0.010
export COMPILE_ENABLED=0
export COMPILE_FULLGRAPH=0

bash experiments/GreenRod_X_1/lab_protocol_20260327/run_ab_1gpu_promote.sh

echo "__RUN_COMPLETE__"
echo "RESULT_DIR=${RESULT_DIR}"
[[ -f "${RESULT_DIR}/promotion_summary.txt" ]] && cat "${RESULT_DIR}/promotion_summary.txt"
