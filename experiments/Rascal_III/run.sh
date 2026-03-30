#!/bin/bash
set -euo pipefail
# Rascal III — TurboMuon + EngramLite combo, full 600s production run
#
# Findings baked in (all in train_gpt.py defaults):
#   TurboMuon:   AOL left-Gram preconditioning + Polar Express NS4 coefficients
#                + row_col post-NS normalize  →  -0.00299 BPB vs baseline
#   EngramLite:  2-head 8192-bucket bigram+trigram hash embedding (2-order)
#                → -0.00006 BPB solo, but -0.00193 extra on top of TurboMuon
#   Combo:       TurboMuon + EngramLite together  →  -0.00492 BPB vs baseline
#
# Arch (matches Rascal II SOTA):
#   11 layers, XSA-all, 512d, 8H/4KV, ROPE_DIMS=16, LATE_QAT=0.15
#
# Usage:
#   bash experiments/Rascal_III/run.sh
#   SEED=300 bash experiments/Rascal_III/run.sh
#   SEED=444 bash experiments/Rascal_III/run.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-42}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p logs

echo "============================================"
echo "  RASCAL III"
echo "  TurboMuon (AOL+NS4+row_col) + EngramLite (8192-bucket 2-head 2-order)"
echo "  Seed: ${SEED}  |  600s wallclock  |  8xH100"
echo "  Expected: ~1.105 BPB (Rascal II 1.1099 - 0.0049 combo delta)"
echo "============================================"

SEED="${SEED}" \
MAX_WALLCLOCK_SECONDS=600 \
LOADER_MODE=coprime \
SKIP_FINAL_EVAL=0 \
POST_EMA_DIAGNOSTIC=0 \
EVAL_STRIDE=64 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
XSA_LAST_N=11 \
ROPE_DIMS=16 \
LATE_QAT_THRESHOLD=0.15 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
NGRAM_BUCKETS=8192 \
NGRAM_HEADS=2 \
NGRAM_ORDERS=2 \
NGRAM_DIM_PER_HEAD=32 \
MUON_BACKEND_STEPS=4 \
MUON_POST_NORM=row_col \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/rascal_iii_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE — copy final_model.pt before reuse"
echo "============================================"
