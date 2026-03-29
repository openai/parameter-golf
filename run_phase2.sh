#!/bin/bash
# Phase 2: Eval-time augmentation ablations.
set -euo pipefail

BASE="${1:?Usage: run_phase2.sh <best_phase1_run_id>}"
export NPROC=1
export MAX_WALLCLOCK_SECONDS=120
export SEED=1337

echo "=========================================="
echo "PHASE 2: EVAL-TIME ABLATIONS"
echo "Base: $BASE"
echo "=========================================="

BASE_TECHS=$(python3 -c "from ablation import load_result; print(','.join(load_result('$BASE').techniques))")

./run_ablation.sh \
  "p2_ttt" \
  "${BASE_TECHS},E2" \
  "$BASE" \
  NOTES="+Legal score-first TTT (AdamW, 3 epochs)" ENABLE_TTT=1

./run_ablation.sh \
  "p2_ttt_tempcal" \
  "${BASE_TECHS},E2,E3" \
  "p2_ttt" \
  NOTES="+Post-TTT temp calibration T=0.98" ENABLE_TTT=1 TTT_TEMP=0.98

./run_ablation.sh \
  "p2_ngram_nottt" \
  "${BASE_TECHS},E6,E7" \
  "$BASE" \
  NOTES="+Multi-order n-gram cache (no TTT)" ENABLE_NGRAM=1

./run_ablation.sh \
  "p2_ngram_knn_nottt" \
  "${BASE_TECHS},E6,E7,E8" \
  "p2_ngram_nottt" \
  NOTES="+kNN-LM on top of n-gram cache" ENABLE_NGRAM=1 ENABLE_KNN=1

./run_ablation.sh \
  "p2_full_stack" \
  "${BASE_TECHS},E2,E3,E6,E7,E8" \
  "p2_ngram_knn_nottt" \
  NOTES="Full eval stack: TTT+tempcal+ngram+kNN" \
  ENABLE_TTT=1 TTT_TEMP=0.98 ENABLE_NGRAM=1 ENABLE_KNN=1

echo ""
echo "=========================================="
echo "PHASE 2 RESULTS"
echo "=========================================="
python3 ablation.py leaderboard
python3 ablation.py ablations
