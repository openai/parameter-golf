#!/bin/bash
# Run all experiments sequentially. Run from anywhere on the pod.
# Usage: bash /workspace/my-parameter-golf/parameter-golf/run_experiments.sh
set -e

BASE=/workspace/my-parameter-golf

# Git identity (needed for commits on fresh pods)
git config --global user.email "runpod@parameter-golf" 2>/dev/null || true
git config --global user.name "FlashyFlash3011" 2>/dev/null || true
DATA=$BASE/data/datasets/fineweb10B_sp1024
TOK=$BASE/data/tokenizers/fineweb_1024_bpe.model
RECORDS=$BASE/records/track_10min_16mb

save_and_push() {
    local dir=$1
    local seed=$2
    local exp_name=$(basename $dir)
    cd $BASE
    git add "$dir/seed${seed}.log" 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "results: ${exp_name} seed${seed}"
        git push fork flashyflash3011/Experiments || \
            echo "WARNING: push failed, log is committed locally"
    fi
}

run_seed() {
    local dir=$1
    local seed=$2
    local extra_env=${3:-""}
    echo ""
    echo "========================================="
    echo "Running: $(basename $dir) | SEED=$seed"
    echo "========================================="
    cd "$dir"
    env SEED=$seed \
        DATA_PATH=$DATA TOKENIZER_PATH=$TOK \
        $extra_env \
        torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee seed${seed}.log
    save_and_push "$dir" "$seed"
}

# --- GPTQLite_QAT_MaxLZMA_LegalTTT (~98ms/step, 11L int6) ---
# GatedAttn + ValueResidual + Full QAT from step 1 + lzma-9 + BigramHash(2048) + Legal TTT (3ep SGD)
# Target: < 1.1144 BPB
EXP=$RECORDS/2026-03-27_GPTQLite_QAT_MaxLZMA_LegalTTT
EXTRA="QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.05 TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=0 ITERATIONS=9000 WARMDOWN_ITERS=3500"
run_seed $EXP 1337 "$EXTRA"
run_seed $EXP 42   "$EXTRA"
run_seed $EXP 2025 "$EXTRA"

echo ""
echo "========================================="
echo "All experiments complete."
echo "========================================="
