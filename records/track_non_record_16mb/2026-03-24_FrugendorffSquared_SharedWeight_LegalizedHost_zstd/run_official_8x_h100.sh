#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

export RUN_ID=${RUN_ID:-approach5_nonrecord_1x_20260325}
export DATA_PATH=${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}
export TOKENIZER_PATH=${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}
export VOCAB_SIZE=${VOCAB_SIZE:-1024}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-4800}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}

export NUM_LAYERS=${NUM_LAYERS:-6}
export NUM_LOOPS=${NUM_LOOPS:-2}
export MODEL_DIM=${MODEL_DIM:-640}
export NUM_HEADS=${NUM_HEADS:-10}
export NUM_KV_HEADS=${NUM_KV_HEADS:-5}
export MLP_MULT=${MLP_MULT:-4}
export BIGRAM_VOCAB_SIZE=${BIGRAM_VOCAB_SIZE:-2048}
export BIGRAM_DIM=${BIGRAM_DIM:-128}
export XSA_LAST_N=${XSA_LAST_N:-2}
export ROPE_DIMS=${ROPE_DIMS:-16}
export VE_ENABLED=${VE_ENABLED:-1}
export VE_DIM=${VE_DIM:-128}
export VE_LAYERS=${VE_LAYERS:-2,3}
export DTG_ENABLED=${DTG_ENABLED:-0}
export LOGIT_SOFTCAP=${LOGIT_SOFTCAP:-30.0}
export TTT_BURST_ENABLED=${TTT_BURST_ENABLED:-0}
export DISTILL_ENABLED=${DISTILL_ENABLED:-0}

python3 -c "import importlib.util, zstandard, sentencepiece, huggingface_hub; assert importlib.util.find_spec('flash_attn') or importlib.util.find_spec('flash_attn_interface')" >/dev/null 2>&1 || {
  echo "dependency_missing: install zstandard, sentencepiece, huggingface_hub, and flash-attn before launch" >&2
  exit 1
}

echo "approach5_launch_start $(date -Iseconds)"
python3 -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT_DIR/train_gpt.py"
echo "approach5_launch_end $(date -Iseconds)"
