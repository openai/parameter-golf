#!/bin/bash
set -e
echo "=== Parameter Golf V6 RunPod Setup ==="
pip install sentencepiece zstandard huggingface_hub 2>/dev/null

# Data setup
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    if [ -d "./datasets/fineweb10B_sp1024" ]; then
        mkdir -p data
        ln -sf "$(pwd)/datasets" data/datasets
        ln -sf "$(pwd)/tokenizers" data/tokenizers
    else
        python3 cached_challenge_fineweb.py --variant sp1024
        mkdir -p data
        ln -sf "$(pwd)/datasets" data/datasets
        ln -sf "$(pwd)/tokenizers" data/tokenizers
    fi
fi
echo "Data ready: $(ls data/datasets/fineweb10B_sp1024/ | wc -l) files"

MODE=${1:-default}
SEED=${SEED:-42}
echo "=== Mode: $MODE | Seed: $SEED ==="

case $MODE in
    smoke)
        # 60-second smoke test — catches crashes before burning a full run ($0.40 vs $8)
        echo "SMOKE TEST: 60s training + quick eval — catching crashes early"
        MAX_WALLCLOCK_SECONDS=60 VAL_LOSS_EVERY=0 NGRAM_EVAL_ORDER=0 \
            SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
        echo "SMOKE TEST PASSED — safe to run full"
        ;;
    default)
        echo "V6: 10L d=512 4KV LeakyReLU^2 XSA4 PartialRoPE VR EMA + 7-gram backoff + entropy-adaptive"
        SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
        ;;
    fast)
        # Smoke test then full run back-to-back
        echo "=== SMOKE TEST (60s) ==="
        MAX_WALLCLOCK_SECONDS=60 VAL_LOSS_EVERY=0 NGRAM_EVAL_ORDER=0 \
            SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
        echo "=== SMOKE PASSED — LAUNCHING FULL RUN ==="
        SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
        ;;
    no_ngram)
        echo "Ablation: no n-gram cache"
        NGRAM_EVAL_ORDER=0 SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt.py
        ;;
    three_seed)
        for S in 42 1337 2024; do
            echo "=== Seed $S ==="
            SEED=$S torchrun --standalone --nproc_per_node=8 train_gpt.py
        done
        ;;
    *)
        echo "Modes: smoke|default|fast|no_ngram|three_seed"
        exit 1
        ;;
esac
echo "=== Done ==="
