#!/bin/bash
set -euo pipefail
# FX-WING MICRO — concept test for GB10 Blackwell DGX Spark
# No CUDA required — works on cuda/mps/cpu
# Tiny model, short run, validates instructed recurrence + CRAWLER_QUANT_INT8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1337}"

echo "============================================"
echo "  FX-WING MICRO — GB10 Blackwell concept test"
echo "  Seed: ${SEED}"
echo "  dim=128 | 2 flat + 1 crawler x 2 loops"
echo "  inst_dim=16 | CRAWLER_QUANT_INT8=1"
echo "  wallclock=120s | single process"
echo "============================================"

SEED="$SEED" \
MAX_WALLCLOCK_SECONDS=120 \
COMPILE_ENABLED=0 \
COMPILE_FULLGRAPH=0 \
DDP_FIND_UNUSED_PARAMETERS=0 \
MODEL_DIM=128 \
NUM_LAYERS=4 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2.0 \
MLP_ACT=relu_sq \
TRAIN_SEQ_LEN=256 \
EVAL_SEQ_LEN=256 \
TRAIN_BATCH_TOKENS=32768 \
ITERATIONS=10000 \
WARMUP_STEPS=10 \
WARMDOWN_ITERS=200 \
GRAD_CLIP_NORM=0.3 \
MATRIX_LR=0.03 \
SCALAR_LR=0.03 \
TIED_EMBED_LR=0.035 \
VAL_LOSS_EVERY=50 \
VAL_BATCH_SIZE=32768 \
EVAL_STRIDE=16 \
SWA_ENABLED=0 \
SWA_EVERY=0 \
QAT_ENABLED=0 \
LATE_QAT_THRESHOLD=0 \
ROPE_DIMS=8 \
BIGRAM_VOCAB_SIZE=512 \
XSA_LAST_N=2 \
MTP_NUM_HEADS=0 \
TRIGRAM=0 \
COMPLEMENT_ALPHA=0 \
NGRAM_EVAL_ORDER=5 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=1 \
NGRAM_EVAL_BUCKETS=1048576 \
NGRAM_EVAL_MAX_SECONDS=30 \
NGRAM_CHUNK_TOKENS=16384 \
CUBRIC_CADENCE=0 \
NGRAM_ENTROPY_SHIFT=0 \
NGRAM_DIRICHLET=1 \
NGRAM_DIRICHLET_CONC=5.0 \
PHRASE_CACHE=0 \
REGIME_TRACKER=0 \
ARTIFACT_NGRAM=0 \
USE_CRAWLER=1 \
NUM_FLAT_LAYERS=2 \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=2 \
CRAWLER_MLP_MULT=2.0 \
INST_DIM=16 \
CRAWLER_QUANT_INT8=1 \
DELTA_NET_HEADS=2 \
python3 -u "${SCRIPT_DIR}/micro_train_gpt.py" \
    2>&1 | tee "logs/fxwing_micro_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  MICRO DONE"
echo "============================================"
