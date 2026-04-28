#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [ "$GPU_COUNT" -ne 8 ]; then
  echo "ERROR: expected 8 visible GPUs, found $GPU_COUNT"
  exit 1
fi

python3 -c "import brotli, sentencepiece, torch; print('torch', torch.__version__)"

if [ ! -f data/tokenizers/fineweb_8192_bpe.model ]; then
  echo "ERROR: missing data/tokenizers/fineweb_8192_bpe.model"
  exit 1
fi

TRAIN_SHARDS="$(find data/datasets/fineweb10B_sp8192 -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l)"
VAL_SHARDS="$(find data/datasets/fineweb10B_sp8192 -maxdepth 1 -name 'fineweb_val_*.bin' 2>/dev/null | wc -l)"
if [ "$TRAIN_SHARDS" -lt 80 ] || [ "$VAL_SHARDS" -lt 1 ]; then
  echo "ERROR: incomplete SP8192 data: train_shards=$TRAIN_SHARDS val_shards=$VAL_SHARDS"
  exit 1
fi

DATA_PATH=data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
NUM_UNIQUE_LAYERS=11 \
NUM_RECURRENCES=3 \
TARGETED_RECURRENCE=1 \
RECURRENCE_START_LAYER=3 \
RECURRENCE_END_LAYER=5 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=4 \
MLP_NEGATIVE_SLOPE=0.5 \
TRAIN_SEQ_LEN=1024 \
TIE_EMBEDDINGS=1 \
QK_GAIN_INIT=5.25 \
ROPE_BASE=10000 \
ROPE_FRACTION=0.25 \
PARALLEL_RESIDUALS=1 \
PARALLEL_LATER_RESIDUALS=1 \
LAYERWISE_NORM_SCALE=1 \
LOGIT_SOFTCAP=30.0 \
MATRIX_LR=0.022 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WEIGHT_DECAY=0.095 \
MUON_ROW_NORM=1 \
GRAD_CLIP_NORM=0.3 \
EXPORT_BITS=6 \
EMBED_EXPORT_BITS=8 \
QUANT_METHOD=gptq \
GPTQ_EMBED=0 \
USE_SDCLIP=1 \
SDCLIP_K=12.85 \
EMA_DECAY=0.9965 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=256 \
TTT_ENABLED=1 \
TTT_LR=5e-3 \
TTT_CHUNK_TOKENS=32768 \
TTT_EPOCHS=3 \
TTT_MAX_CHUNKS=0 \
TTT_ADAPT_ENABLED=0 \
CTRL_SURFACE_LAMBDA=0.1 \
COMPRESS_METHOD=brotli \
BYTE_SHUFFLE_STRIDE=2 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_FRAC=0.72 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
RUN_ID=submission_frontier \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04_tns15june_v1/train_gpt.py \
  2>&1 | tee logs/submission_frontier.console.txt
