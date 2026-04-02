#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TRAINER="${REPO_ROOT}/junkyard/experiments/Rascal_Final_Submission_LC4/train_gpt.py"
DATA_PATH="${REPO_ROOT}/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model"
LOG_DIR="${REPO_ROOT}/logs"

cd "${REPO_ROOT}"

die() { echo "FATAL: $*" >&2; exit 1; }

if [[ -f "${REPO_ROOT}/scripts/activate_flywheel_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/activate_flywheel_env.sh"
elif [[ -x /workspace/miniconda3/bin/conda && -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-fa3wheel}" >/dev/null 2>&1 || true
elif [[ -f "${VENV_DIR:-/workspace/venv_cu124}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR:-/workspace/venv_cu124}/bin/activate"
elif [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

PYTHON_BIN="$(command -v python || true)"
[[ -n "${PYTHON_BIN}" ]] || die "python not found after env activation"
"${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1 || die "python cannot import torch; activate the pod env first"

export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-444}"
mkdir -p "${LOG_DIR}"
RUN_ID="rascal_lc4_s${SEED}_$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${RUN_ID}.log"

SEED="${SEED}" \
RUN_ID="${RUN_ID}" \
DATA_PATH="${DATA_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=500 \
COMPILE_ENABLED=1 \
COMPILE_FULLGRAPH=1 \
SKIP_GPTQ=1 \
LOADER_MODE=coprime \
COPRIME_MAX_LOADED_SHARDS=4 \
COPRIME_SHARDS_PER_BATCH=1 \
COPRIME_SHARD_HOLD_STEPS=64 \
COMPLEMENT_ALPHA=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
NGRAM_EVAL_ORDER=0 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
"${TRAINER}" \
2>&1 | tee "${LOG}"
