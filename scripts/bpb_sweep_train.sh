#!/usr/bin/env bash
# Phase 1c: training hyperparameter grid (WD, MLR, EMA, warmdown, grad clip).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TRAIN="${TRAIN_SCRIPT:-train_gpt_from_blob.py}"
NPROC="${NPROC_PER_NODE:-8}"
for MUON_WD in 0.08 0.095 0.11; do
  for MATRIX_LR in 0.018 0.022 0.026; do
    for EMA_DECAY in 0.996 0.9965 0.997; do
      for WARMDOWN_FRAC in 0.68 0.72 0.76; do
        for GRAD_CLIP_NORM in 0.3 0.5 1.0; do
          echo "========== MUON_WD=$MUON_WD MATRIX_LR=$MATRIX_LR EMA=$EMA_DECAY WD_FRAC=$WARMDOWN_FRAC CLIP=$GRAD_CLIP_NORM =========="
          MUON_WD="$MUON_WD" MATRIX_LR="$MATRIX_LR" EMA_DECAY="$EMA_DECAY" \
            WARMDOWN_FRAC="$WARMDOWN_FRAC" GRAD_CLIP_NORM="$GRAD_CLIP_NORM" \
            torchrun --standalone --nproc_per_node="$NPROC" "$TRAIN" || true
        done
      done
    done
  done
done
