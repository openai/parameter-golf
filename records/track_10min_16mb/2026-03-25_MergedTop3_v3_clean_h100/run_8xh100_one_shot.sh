#!/usr/bin/env bash
set -euo pipefail

bash preflight_remote_strict.sh
source "${REMOTE_VENV_DIR:-/workspace/.venvs/parameter-golf-20260325}/bin/activate"

export PYTHONUNBUFFERED=1
export RUN_ID="${RUN_ID:-mergedtop3_v3_8xh100_clean}"
export SEED="${SEED:-1337}"
export DATA_PATH="${DATA_PATH:-../../../data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-../../../data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export WALLCLOCK_BUFFER_SECONDS="${WALLCLOCK_BUFFER_SECONDS:-20}"
export ITERATIONS="${ITERATIONS:-9000}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
export EVAL_STRIDE="${EVAL_STRIDE:-64}"
export AUTO_RESUME="${AUTO_RESUME:-0}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-200}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoint_latest.pt}"

export NUM_LAYERS="${NUM_LAYERS:-11}"
export XSA_LAST_N="${XSA_LAST_N:-4}"
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_DECAY="${EMA_DECAY:-0.997}"
export QAT_ENABLED="${QAT_ENABLED:-0}"
export SWA_ENABLED="${SWA_ENABLED:-0}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export ROPE_DIMS="${ROPE_DIMS:-16}"
export LN_SCALE="${LN_SCALE:-1}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3500}"
export MATRIX_LR="${MATRIX_LR:-0.025}"
export SCALAR_LR="${SCALAR_LR:-0.025}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export MUON_WD="${MUON_WD:-0.04}"
export ADAM_WD="${ADAM_WD:-0.04}"

python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "train_seed${SEED}.log"
