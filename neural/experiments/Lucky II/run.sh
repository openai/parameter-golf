#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}/flash-attention/hopper${PYTHONPATH:+:${PYTHONPATH}}"
export SEED="${SEED:-300}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)}"
if [[ -z "${NPROC_PER_NODE}" || "${NPROC_PER_NODE}" == "0" ]]; then
  NPROC_PER_NODE=1
fi

python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${ROOT}/neural/experiments/Lucky II/modify_me.py"
