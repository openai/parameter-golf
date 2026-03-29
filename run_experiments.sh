#!/bin/bash
# Official submission run for GPTQLite_QAT_MaxLZMA_LegalTTT
# Usage: bash /workspace/my-parameter-golf/run_experiments.sh
set -e

BASE=/workspace/my-parameter-golf

# Git identity (needed for commits on fresh pods)
git config --global user.email "runpod@parameter-golf" 2>/dev/null || true
git config --global user.name "FlashyFlash3011" 2>/dev/null || true
DATA=$BASE/data/datasets/fineweb10B_sp1024
TOK=$BASE/data/tokenizers/fineweb_1024_bpe.model
EXP=$BASE/records/track_10min_16mb/2026-03-27_GPTQLite_QAT_MaxLZMA_LegalTTT

save_and_push() {
    local seed=$1
    cd $BASE
    git add "$EXP/seed${seed}.log" 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "results: GPTQLite_QAT_MaxLZMA_LegalTTT seed${seed}"
        git push fork flashyflash3011/Experiments || \
            echo "WARNING: push failed, log is committed locally"
    fi
}

run_seed() {
    local seed=$1
    echo ""
    echo "========================================="
    echo "Running: GPTQLite_QAT_MaxLZMA_LegalTTT | SEED=$seed"
    echo "========================================="
    cd "$EXP"
    NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 GATED_ATTENTION=0 VALUE_RESIDUAL=0 \
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 ROPE_DIMS=16 LN_SCALE=1 \
    MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
    LATE_QAT_THRESHOLD=0.15 BANK_QAT_THRESHOLD=0 SWA_ENABLED=0 \
    TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 \
    ITERATIONS=9000 WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=600 \
    DATA_PATH=$DATA TOKENIZER_PATH=$TOK SEED=$seed \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee seed${seed}.log
    save_and_push "$seed"
}

run_seed 1337
run_seed 42
run_seed 2025

echo ""
echo "========================================="
echo "All seeds complete."
echo "========================================="
