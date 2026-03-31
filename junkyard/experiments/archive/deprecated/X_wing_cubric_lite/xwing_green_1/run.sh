#!/usr/bin/env bash
set -euo pipefail
# X-wing Green 1: PR#779 BackoffNgramMixer + Cubric per-order adaptive alpha
# Cubric settings: proven green config (floor=0.3, ceiling=2.0, adapt=1.03/0.97)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/environment/vars.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/environment/vars.env"
  set +a
fi

: "${SEED:=1337}"
: "${MAX_WALLCLOCK_SECONDS:=600}"
: "${NPROC_PER_NODE:=8}"
: "${PYTHON_BIN:=python3}"

export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-xwing_green1_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

# Cubric per-order adaptive alpha scaling (proven green config)
export CUBRIC_ENABLED=1
export CUBRIC_FLOOR=0.3
export CUBRIC_CEILING=2.0
export CUBRIC_ADAPT_UP=1.03
export CUBRIC_ADAPT_DOWN=0.97
export CUBRIC_ALPHA_CLIP=0.70

# Kill TTT — adds only 0.005 BPB but doubles eval time
export TTT_EPOCHS=0

echo "============================================"
echo "  X-WING GREEN 1 (cubric per-order scaling)"
echo "  Seed: ${SEED}"
echo "  Cubric: floor=${CUBRIC_FLOOR} ceil=${CUBRIC_CEILING} clip=${CUBRIC_ALPHA_CLIP}"
echo "============================================"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
