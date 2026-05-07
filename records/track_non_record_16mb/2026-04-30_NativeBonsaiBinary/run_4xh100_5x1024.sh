#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC="${NPROC:-4}"
OUT_DIR="${OUT_DIR:-$ROOT/logs_4xh100_5x1024}"
RUN_ID="${RUN_ID:-4xh100_5x1024_rankoffset_time590_rerun}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
MATCHED_FINEWEB_REPO_ID="${MATCHED_FINEWEB_REPO_ID:-kevclark/parameter-golf}"

mkdir -p "$OUT_DIR"

echo "== 4xH100 grouped-binary 5x1024 run =="
echo "root=$ROOT"
echo "nproc=$NPROC"
echo "out_dir=$OUT_DIR"
echo "run_id=$RUN_ID"
echo "train_shards=$TRAIN_SHARDS"
echo "matched_fineweb_repo_id=$MATCHED_FINEWEB_REPO_ID"

"$PYTHON_BIN" -m pip install -q sentencepiece huggingface-hub numpy tqdm

MATCHED_FINEWEB_REPO_ID="$MATCHED_FINEWEB_REPO_ID" \
  "$PYTHON_BIN" data/cached_challenge_fineweb.py --variant sp8192 --train-shards "$TRAIN_SHARDS"

env \
  PYTHONUNBUFFERED=1 \
  DATA_PATH="$ROOT/data/datasets/fineweb10B_sp8192" \
  TOKENIZER_PATH="$ROOT/data/tokenizers/fineweb_8192_bpe.model" \
  RUN_ID="$RUN_ID" \
  OUT_DIR="$OUT_DIR" \
  DDP=1 \
  VOCAB_SIZE=8192 \
  EMBED_DIM=254 \
  NUM_LAYERS=5 \
  MODEL_DIM=1024 \
  NUM_HEADS=16 \
  NUM_KV_HEADS=4 \
  MLP_MULT=4.0 \
  ITERATIONS="${ITERATIONS:-20000}" \
  WARMUP_STEPS=5 \
  WARMDOWN_ITERS="${WARMDOWN_ITERS:-8000}" \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-32768}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
  TRAIN_SEQ_LEN=1024 \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}" \
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}" \
  VAL_LOSS_EVERY=0 \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-590.0}" \
  TIE_EMBEDDINGS=1 \
  QUANT_MODE=binary \
  QUANT_GROUP_SIZE=128 \
  QUANTIZE_EMBEDDINGS=0 \
  BINARY_CENTER_MODE=none \
  SKIP_ROUNDTRIP_EVAL=1 \
  SKIP_FINAL_VAL=1 \
  SAVE_DEBUG_ZLIB=0 \
  TTT_ENABLED=0 \
  OPTIMIZER_NAME=split_muon \
  MATRIX_LR="${MATRIX_LR:-0.006}" \
  SCALAR_LR="${SCALAR_LR:-0.006}" \
  TIED_EMBED_LR="${TIED_EMBED_LR:-0.009}" \
  MUON_MOMENTUM=0.99 \
  MUON_MOMENTUM_WARMUP_START=0.92 \
  MUON_MOMENTUM_WARMUP_STEPS=500 \
  MUON_BACKEND_STEPS=5 \
  ROPE_DIM=16 \
  LOGIT_SOFTCAP=30 \
  SOFTCAP_MODE=tanh \
  ROPE_BASE=1000000 \
  LAYER_SCHEDULE="" \
  PARALLEL_RESIDUAL_START_LAYER=10000 \
  QK_GAIN="${QK_GAIN:-1.0}" \
  MLP_ACT=swiglu \
  torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py \
  2>&1 | tee "$OUT_DIR/${RUN_ID}.txt"

FINAL_CKPT="$OUT_DIR/${RUN_ID}_mlx_model.npz"
if [[ ! -f "$FINAL_CKPT" ]]; then
  echo "ERROR: checkpoint was not created: $FINAL_CKPT" >&2
  exit 2
fi

"$PYTHON_BIN" export_native_packet_size.py "$FINAL_CKPT" \
  --mode binary --group-size 128 --code train_gpt.py \
  | tee "$OUT_DIR/${RUN_ID}_packet_size.txt"

echo "== Done =="
echo "log: $OUT_DIR/${RUN_ID}.txt"
echo "checkpoint: $FINAL_CKPT"
echo "packet size: $OUT_DIR/${RUN_ID}_packet_size.txt"
