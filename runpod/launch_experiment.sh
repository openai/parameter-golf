#!/bin/bash
# Kick off a single training+eval run. Assumes bootstrap.sh has been executed
# at least once on the current pod and /workspace is a persistent volume.
#
# Usage:
#   ./launch_experiment.sh <submission_folder> <seed> [NGRAM_MIX_ENABLED=1] [extra env=val]...
#
# Examples (Day 3 sweep — eight-GPU eval-only — we re-use one trained artifact):
#   ./launch_experiment.sh records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix 42  NGRAM_MIX_ENABLED=1 NGRAM_MIX_ALPHA=2.0 NGRAM_MIX_BETA=-0.25
#   ./launch_experiment.sh records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix 42  NGRAM_MIX_ENABLED=1 NGRAM_MIX_ALPHA=1.0 NGRAM_MIX_BETA=-0.30
#   ...
#
# Outputs: /workspace/runs/<RUN_ID>/{train_gpt.out,model.bin,meta.json}

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <submission_folder_relative_to_repo> <seed> [KEY=VAL]..."
  exit 1
fi

SUB_FOLDER=$1
SEED=$2
shift 2

WS=/workspace
REPO=$WS/repo/parameter-golf
cd "$REPO"

SUB_DIR=$REPO/$SUB_FOLDER
if [ ! -f "$SUB_DIR/train_gpt.py" ]; then
  echo "no train_gpt.py found at $SUB_DIR"
  exit 1
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)_$(basename "$SUB_FOLDER")_s${SEED}
RUN_DIR=$WS/runs/$RUN_ID
mkdir -p "$RUN_DIR"

# Core env. Extend the SP8192 defaults; add a Network-Volume-friendly tokenizer/data path.
export TORCHINDUCTOR_CACHE_DIR=$WS/cache/torch_inductor
export TRITON_CACHE_DIR=$WS/cache/triton
export HF_HOME=$WS/cache/hf
export TOKENIZERS_PARALLELISM=false

export RUN_ID=$RUN_ID
export SEED=$SEED
export DATA_PATH=$WS/data/datasets/fineweb10B_sp8192/
export TOKENIZER_PATH=$WS/data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export QUANTIZED_MODEL_PATH=$RUN_DIR/model.bin

# Apply extra KEY=VAL overrides from CLI.
for kv in "$@"; do
  export "$kv"
done

# Save a meta snapshot before kicking torchrun.
{
  echo "run_id: $RUN_ID"
  echo "submission: $SUB_FOLDER"
  echo "seed: $SEED"
  echo "git_sha: $(git rev-parse HEAD)"
  echo "git_status: $(git status --porcelain | wc -l) dirty files"
  echo "gpu_count: $(nvidia-smi -L | wc -l)"
  echo "env_extras:"
  for kv in "$@"; do echo "  $kv"; done
} >"$RUN_DIR/meta.txt"

# 8-GPU torchrun. Fall back to nproc_per_node=nvidia-smi -L count for
# single-node pods with fewer GPUs.
NGPU=$(nvidia-smi -L | wc -l)
echo "[launch] RUN_ID=$RUN_ID nproc=$NGPU submission=$SUB_FOLDER"
cd "$SUB_DIR"
torchrun --standalone --nproc_per_node=$NGPU train_gpt.py 2>&1 | tee "$RUN_DIR/train_gpt.out"
EXITCODE=${PIPESTATUS[0]}
echo "[launch] done exit=$EXITCODE log=$RUN_DIR/train_gpt.out"
exit $EXITCODE
