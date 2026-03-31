#!/bin/bash
set -euo pipefail
# B-WING II: Cubric ON + entropy shift + alpha fix + fast TTT
# Best of both worlds: our cubric 3D + #809 entropy/alpha + our sliding TTT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "============================================"
echo "  B-WING II — Cubric + Entropy Shift + TTT"
echo "  Seed: ${SEED}"
echo "  Cubric 3D ON + entropy shift + clip 0.95"
echo "  Fast TTT: 1 epoch, SGD, freeze 2 blocks"
echo "  Eval alpha: 0.05-0.60 clip=0.95 | Orders: 2-9"
echo "============================================"

SEED="$SEED" \
F1_CORR_RANK=0 \
DISTILL_ENABLED=0 \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
XSA_LAST_N=4 \
BIGRAM_VOCAB_SIZE=1536 \
ROPE_DIMS=24 \
VAL_LOSS_EVERY=20000 \
TRAIN_LOG_EVERY=1000 \
SWA_EVERY=100 \
COMPLEMENT_ALPHA=0.5 \
NGRAM_EVAL_ORDER=9 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ENTROPY_CENTER=3.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_BUCKETS=8388608 \
NGRAM_EVAL_MAX_SECONDS=0 \
NGRAM_ENTROPY_SHIFT=1 \
CUBRIC_CADENCE=32 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=1 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=2 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
COMPILE_FULLGRAPH=0 \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/train_gpt.py" \
    2>&1 | tee "logs/bwing_II_s${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================"
echo "  DONE"
echo "============================================"
