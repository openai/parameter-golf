#!/bin/bash
# run.sh — runs one or more training modes in sequence.
# Usage: bash run.sh <mode1> [mode2] ...
#
# FULL ABLATION MATRIX:
#
#   mode          | JEPA | BigramHash | LeakyReLU | purpose
#   --------------|------|------------|-----------|------------------------------
#   baseline      |  0   |     0      |     0     | zero baseline, identical to v1
#   leaky         |  0   |     0      |     1     | isolates LeakyReLU contribution
#   bigram        |  0   |     1      |     1     | isolates BigramHash (+leaky)
#   jepa          |  1   |     0      |     1     | isolates JEPA v2 (+leaky)
#   full          |  1   |     1      |     1     | full stack
#   smoke         |  1   |     1      |     1     | 2-min smoke test
#
# Run full matrix (5 runs x 10 min = ~50 min):
#   bash run.sh baseline leaky bigram jepa full
#
# Quick comparison (main contrast only):
#   bash run.sh baseline full

set -e

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TOKENIZER="$ROOT/data/tokenizers/fineweb_1024_bpe.model"
DATA="$ROOT/data/datasets/fineweb10B_sp1024"

COMMON_FULL="
  MAX_WALLCLOCK_SECONDS=600
  TRAIN_BATCH_TOKENS=131072
  VAL_LOSS_EVERY=200
  TRAIN_LOG_EVERY=50
  ARTIFACT_EMA_DECAY=0.99
  QUANT_MAX=31
  TOKENIZER_PATH=$TOKENIZER DATA_PATH=$DATA
"

if [ $# -eq 0 ]; then
  echo "Usage: $0 [smoke|full|jepa|bigram|leaky|baseline] ..."
  echo ""
  echo "  baseline  10 min — pure CE, no modifications (zero baseline)"
  echo "  leaky     10 min — + LeakyReLU(0.5)^2"
  echo "  bigram    10 min — + LeakyReLU + BigramHash2048"
  echo "  jepa      10 min — + LeakyReLU + JEPA v2 (no BigramHash)"
  echo "  full      10 min — full stack (JEPA + BigramHash + LeakyReLU)"
  echo "  smoke      2 min — smoke test, full stack"
  echo ""
  echo "Full matrix: bash $0 baseline leaky bigram jepa full"
  exit 1
fi

run_mode() {
  local MODE="$1"
  case "$MODE" in

    baseline)
      echo ""
      echo "======================================================"
      echo "  BASELINE — pure CE, ReLU^2, no JEPA, no BigramHash"
      echo "  (zero baseline: comparable to v1)"
      echo "  $(date)"
      echo "======================================================"
      USE_JEPA=0 \
      BIGRAM_VOCAB_SIZE=0 \
      MLP_LEAKY_SLOPE=0.0 \
      MAX_WALLCLOCK_SECONDS=600 TRAIN_BATCH_TOKENS=131072 \
      VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
      ARTIFACT_EMA_DECAY=0.99 QUANT_MAX=31 \
      TOKENIZER_PATH="$TOKENIZER" DATA_PATH="$DATA" \
      python3 -u train_gpt.py
      ;;

    leaky)
      echo ""
      echo "======================================================"
      echo "  LEAKY — LeakyReLU(0.5)^2 only, no JEPA, no BigramHash"
      echo "  Isolates LeakyReLU contribution"
      echo "  $(date)"
      echo "======================================================"
      USE_JEPA=0 \
      BIGRAM_VOCAB_SIZE=0 \
      MLP_LEAKY_SLOPE=0.5 \
      MAX_WALLCLOCK_SECONDS=1800 TRAIN_BATCH_TOKENS=131072 \
      VAL_LOSS_EVERY=400 TRAIN_LOG_EVERY=50 \
      ARTIFACT_EMA_DECAY=0.99 QUANT_MAX=31 \
      TOKENIZER_PATH="$TOKENIZER" DATA_PATH="$DATA" \
      python3 -u train_gpt.py
      ;;

    bigram)
      echo ""
      echo "======================================================"
      echo "  BIGRAM — LeakyReLU + BigramHash2048, no JEPA"
      echo "  Isolates BigramHash contribution"
      echo "  $(date)"
      echo "======================================================"
      USE_JEPA=0 \
      BIGRAM_VOCAB_SIZE=2048 \
      MLP_LEAKY_SLOPE=0.5 \
      MAX_WALLCLOCK_SECONDS=600 TRAIN_BATCH_TOKENS=131072 \
      VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
      ARTIFACT_EMA_DECAY=0.99 QUANT_MAX=31 \
      TOKENIZER_PATH="$TOKENIZER" DATA_PATH="$DATA" \
      python3 -u train_gpt.py
      ;;

    jepa)
      echo ""
      echo "======================================================"
      echo "  JEPA — LeakyReLU + JEPA v2 multi-step, no BigramHash"
      echo "  Isolates JEPA contribution (momentum=0.9, offsets 1,2,4,8)"
      echo "  $(date)"
      echo "======================================================"
      USE_JEPA=1 JEPA_LAMBDA=0.12 \
      JEPA_EMA_MOMENTUM=0.9 \
      BIGRAM_VOCAB_SIZE=0 \
      MLP_LEAKY_SLOPE=0.5 \
      MAX_WALLCLOCK_SECONDS=1800 TRAIN_BATCH_TOKENS=131072 \
      VAL_LOSS_EVERY=400 TRAIN_LOG_EVERY=50 \
      ARTIFACT_EMA_DECAY=0.99 QUANT_MAX=31 \
      TOKENIZER_PATH="$TOKENIZER" DATA_PATH="$DATA" \
      python3 -u train_gpt.py
      ;;

    full)
      echo ""
      echo "======================================================"
      echo "  FULL — full stack"
      echo "  JEPA v2 + BigramHash2048 + LeakyReLU + int6+LZMA + EMA artifact"
      echo "  $(date)"
      echo "======================================================"
      USE_JEPA=1 JEPA_LAMBDA=0.12 \
      JEPA_EMA_MOMENTUM=0.9 \
      BIGRAM_VOCAB_SIZE=2048 \
      MLP_LEAKY_SLOPE=0.5 \
      MAX_WALLCLOCK_SECONDS=600 TRAIN_BATCH_TOKENS=131072 \
      VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
      ARTIFACT_EMA_DECAY=0.99 QUANT_MAX=31 \
      TOKENIZER_PATH="$TOKENIZER" DATA_PATH="$DATA" \
      python3 -u train_gpt.py
      ;;

    smoke)
      echo ""
      echo "======================================================"
      echo "  SMOKE TEST — 2 min, full stack"
      echo "  $(date)"
      echo "======================================================"
      USE_JEPA=1 JEPA_LAMBDA=0.12 \
      JEPA_EMA_MOMENTUM=0.9 \
      BIGRAM_VOCAB_SIZE=2048 \
      MLP_LEAKY_SLOPE=0.5 \
      MAX_WALLCLOCK_SECONDS=120 TRAIN_BATCH_TOKENS=65536 \
      VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=10 \
      ARTIFACT_EMA_DECAY=0.99 QUANT_MAX=31 \
      TOKENIZER_PATH="$TOKENIZER" DATA_PATH="$DATA" \
      python3 -u train_gpt.py
      ;;

    *)
      echo "Unknown mode: '$MODE'"
      echo "Valid modes: smoke | full | jepa | bigram | leaky | baseline"
      exit 1
      ;;
  esac
}

TOTAL=$#
IDX=0
for MODE in "$@"; do
  IDX=$((IDX + 1))
  echo ""
  echo ">>> Run $IDX/$TOTAL: $MODE"
  run_mode "$MODE"
  echo ">>> Done: $MODE ($(date))"
done

echo ""
echo "======================================================"
echo "  ALL RUNS COMPLETED ($TOTAL/$TOTAL)"
echo "  $(date)"
echo ""
echo "  To read results:"
echo "  grep 'val_bpb\|use_jepa\|bigram_params\|mlp_leaky\|int6' logs/*.txt"
echo "======================================================"
