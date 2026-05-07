#!/bin/bash
# Parameter Golf — Phase 1b: SP4096 + Depth Recurrence + Parallel Residuals + QK-Gain + Brotli
# 3-seed validated 2026-04-05: mean 1.1020 BPB (seeds 42, 314, 999)
# Usage: bash run_pgolf_phase1b.sh [seed]

SEED="${1:-42}"

# === SP4096 ARCHITECTURE ===
export VOCAB_SIZE=4096
export MLP_MULT=4.0
export XSA_LAST_N=11

# === PR #1334 TECHNIQUES ===
export QK_GAIN_INIT=5.0
export MUON_EQ_R=1
export RECUR_LAYERS="4,5"
export RECUR_START_STEP=3000
export PARALLEL_START_LAYER=7

# === OPTIMIZER ===
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export TIED_EMBED_LR=0.03
export MUON_WD=0.090
export ADAM_WD=0.090
export WARMDOWN_ITERS=4000

# === GPTQ (AR self-gen, Brotli) ===
export GPTQ_CALIB_BATCHES=128
export GPTQ_DAMP=0.01
export LATE_QAT_THRESHOLD=0.15

# === DISABLED (don't fit SP4096 or confirmed dead) ===
export BIGRAM_VOCAB_SIZE=0
export BIGRAM_DIM=0
export TRIGRAM=0
export HADAMARD_ROTATION=0
export SOFT_ROUND_QAT=0
export PREQUANT_TTT=0
export MIXED_BITWIDTH=0
export NGRAM_ENABLED=0
export TTT_ENABLED=0

# === DATA PATHS (SP4096) ===
export DATA_PATH=./data/datasets/fineweb10B_sp4096
export TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model

# === PROVEN STACK ===
export SWA_ENABLED=1
export VE_ENABLED=1
export LN_SCALE=1
export TARGET_MB=15.9
export SEED=$SEED

# === FULL RUN (10 min wallclock) ===
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=4000

echo "=== Phase 1b: SP4096 + DepthRecur + ParallelResid + QK-Gain + Brotli ==="
echo "Seed: $SEED | Vocab: $VOCAB_SIZE | MLP: ${MLP_MULT}x | ADAM_WD: $ADAM_WD"
echo "DepthRecur: layers $RECUR_LAYERS from step $RECUR_START_STEP"
echo "ParallelResid: from layer $PARALLEL_START_LAYER | QK-Gain: $QK_GAIN_INIT"
echo ""

torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "run_phase1b_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
