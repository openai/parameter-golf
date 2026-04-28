#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)}"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" == "0" ]]; then
  NPROC_PER_NODE=1
fi

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"
export SEED="${SEED:-300}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-4}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-2000}"
export WARMDOWN_MODE="${WARMDOWN_MODE:-linear}"
export SKIP_GPTQ="${SKIP_GPTQ:-1}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT}/neural/experiments/QK4_Contender/train_gpt.py"
