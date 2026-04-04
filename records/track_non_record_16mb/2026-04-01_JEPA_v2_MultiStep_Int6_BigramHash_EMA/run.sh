#!/bin/bash
# run.sh — lancia uno o più run in sequenza.
# Uso: bash run.sh <mode1> [mode2] ...
#
# MATRICE DI ABLATION COMPLETA:
#
#   mode          | JEPA | BigramHash | LeakyReLU | scopo
#   --------------|------|------------|-----------|------------------------------
#   baseline      |  0   |     0      |     0     | punto zero, identico v1
#   leaky         |  0   |     0      |     1     | isola LeakyReLU
#   bigram        |  0   |     1      |     1     | isola BigramHash (+leaky)
#   jepa          |  1   |     0      |     1     | isola JEPA v2 (+leaky)
#   full          |  1   |     1      |     1     | stack completo
#   smoke         |  1   |     1      |     1     | smoke test 2 min
#
# Eseguire la matrice completa (5 run da 10 min = ~50 min):
#   bash run.sh baseline leaky bigram jepa full
#
# Run rapido con il confronto principale:
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
  echo "Uso: $0 [smoke|full|jepa|bigram|leaky|baseline] ..."
  echo ""
  echo "  baseline  10 min — CE puro, nessuna modifica (punto zero)"
  echo "  leaky     10 min — + LeakyReLU(0.5)^2"
  echo "  bigram    10 min — + LeakyReLU + BigramHash2048"
  echo "  jepa      10 min — + LeakyReLU + JEPA v2 (no BigramHash)"
  echo "  full      10 min — stack completo (JEPA + BigramHash + LeakyReLU)"
  echo "  smoke      2 min — smoke test stack completo"
  echo ""
  echo "Matrice completa: bash $0 baseline leaky bigram jepa full"
  exit 1
fi

run_mode() {
  local MODE="$1"
  case "$MODE" in

    baseline)
      echo ""
      echo "======================================================"
      echo "  BASELINE — CE puro, ReLU^2, no JEPA, no BigramHash"
      echo "  (punto zero: confrontabile con v1)"
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
      echo "  LEAKY — solo LeakyReLU(0.5)^2, no JEPA, no BigramHash"
      echo "  Isola il contributo di LeakyReLU"
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
      echo "  Isola il contributo di BigramHash"
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
      echo "  Isola il contributo di JEPA (momentum=0.9, offsets 1,2,4,8)"
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
      echo "  FULL — stack completo"
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
      echo "  SMOKE TEST — 2 min, stack completo"
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
      echo "Modalità sconosciuta: '$MODE'"
      echo "Valide: smoke | full | jepa | bigram | leaky | baseline"
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
  echo ">>> Completato: $MODE ($(date))"
done

echo ""
echo "======================================================"
echo "  TUTTI I RUN COMPLETATI ($TOTAL/$TOTAL)"
echo "  $(date)"
echo ""
echo "  Per leggere i risultati:"
echo "  grep 'val_bpb\|use_jepa\|bigram_params\|mlp_leaky\|int6' logs/*.txt"
echo "======================================================"
