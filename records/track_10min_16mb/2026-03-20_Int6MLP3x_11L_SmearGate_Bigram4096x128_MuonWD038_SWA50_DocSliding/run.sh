#!/usr/bin/env bash

set -euo pipefail

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RECORD_DIR/../../.." && pwd)"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$RECORD_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS="${NUM_LAYERS:-11}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-3}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
export MATRIX_LR="${MATRIX_LR:-0.02}"
export SCALAR_LR="${SCALAR_LR:-0.02}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
export MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
export MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.038}"
export ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-0.01}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.3}"
export QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
export SMEARGATE_ENABLED="${SMEARGATE_ENABLED:-1}"
export BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-4096}"
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export SWA_EVERY="${SWA_EVERY:-50}"
export SWA_START_FRAC="${SWA_START_FRAC:-0.50}"
export LOWBIT_BITS="${LOWBIT_BITS:-6}"
export LOWBIT_NAME_PATTERNS="${LOWBIT_NAME_PATTERNS:-.mlp.,.attn.c_q.,.attn.c_k.,.attn.c_v.,.attn.proj.}"
export INT8_KEEP_FLOAT_NAME_PATTERNS="${INT8_KEEP_FLOAT_NAME_PATTERNS:-tok_emb.weight,bigram.embed.weight,bigram.proj.weight}"
export INT8_GROUP_OVERRIDES="${INT8_GROUP_OVERRIDES:-.attn.c_k.:64}"
export SERIAL_COMPRESSOR="${SERIAL_COMPRESSOR:-zstd}"
export RUN_ID="${RUN_ID:-record_11l_lexical4096x128_muwd038_swa50_a}"

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py
