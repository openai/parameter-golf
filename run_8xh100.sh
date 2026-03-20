#!/bin/bash
# Parameter Golf - 8xH100 Validation Script
# Based on PR #198 (SOTA 1.1318) with FA3 fallback
# Usage: bash run_8xh100.sh

set -e

echo "=== Parameter Golf 8xH100 Validation ==="
echo "Start: $(date)"

cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/alertcat/parameter-golf.git
fi
cd parameter-golf

# Download data if not present
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "=== Downloading data (80 shards) ==="
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
fi

echo "=== Data ready ==="
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "GPUs: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

N_GPUS=$(python3 -c 'import torch; print(torch.cuda.device_count())')

# Common env vars (PR #198 config with LR=0.025)
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export ITERATIONS=9000
export MAX_WALLCLOCK_SECONDS=600
export WARMDOWN_ITERS=3000
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WEIGHT_DECAY=0.04
export QAT=1
export QUANT_BITS=6
export FP16_EMBED=1
export SMEAR_GATE=1
export BIGRAM_HASH=1
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
export SWA_ENABLED=1
export SWA_START_FRAC=0.5
export SWA_EVERY=200
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=32
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100

for SEED_VAL in 1337 42 2024; do
    echo ""
    echo "======================================================================"
    echo "=== SEED $SEED_VAL | $(date) ==="
    echo "======================================================================"
    export SEED=$SEED_VAL
    export RUN_ID="final_s${SEED_VAL}"

    torchrun --standalone --nproc_per_node=$N_GPUS train_gpt.py 2>&1 | tee "run_seed${SEED_VAL}.txt"

    echo "--- Seed $SEED_VAL complete ---"
    tail -5 "run_seed${SEED_VAL}.txt"
done

echo ""
echo "======================================================================"
echo "=== ALL SEEDS COMPLETE ==="
echo "======================================================================"
echo ""
for f in run_seed*.txt; do
    echo "--- $f ---"
    grep "final_int6_zstd.*val_bpb\|Total submission size\|sliding.*val_bpb" "$f" | tail -3
done
echo ""
echo "End: $(date)"
