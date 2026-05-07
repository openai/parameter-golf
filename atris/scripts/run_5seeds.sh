#!/bin/bash
# Run 5 seeds for statistical significance (required for submission)
# Must show p < 0.01 that improvement >= 0.005 nats over SOTA
#
# Usage: bash atris/scripts/run_5seeds.sh [variant]

set -euo pipefail

cd "$(dirname "$0")/../.."

VARIANT=${1:-sp1024}
NPROC=${NPROC:-8}

echo "=== Running 5 seeds for statistical validation ==="
echo "Variant: $VARIANT"
echo ""

RESULTS=()

for SEED in 1337 1338 1339 1340 1341; do
    echo "--- SEED=$SEED ---"

    NCCL_IB_DISABLE=1 \
    RUN_ID="atris_seed${SEED}_$(date +%s)" \
    SEED=$SEED \
    DATA_PATH="./data/datasets/fineweb10B_${VARIANT}/" \
    TOKENIZER_PATH="./data/tokenizers/fineweb_${VARIANT#sp}_bpe.model" \
    VOCAB_SIZE=${VARIANT#sp} \
    NUM_LAYERS=10 \
    MATRIX_LR=0.02 \
    SCALAR_LR=0.02 \
    TIED_EMBED_LR=0.03 \
    MUON_MOMENTUM=0.99 \
    MUON_MOMENTUM_WARMUP_START=0.92 \
    MLP_MULT=3 \
    EVAL_STRIDE=64 \
    MAX_WALLCLOCK_SECONDS=600 \
    VAL_LOSS_EVERY=0 \
    TRAIN_LOG_EVERY=200 \
    torchrun --standalone --nproc_per_node=$NPROC train_gpt.py 2>&1 | \
        tee "atris/logs/seed_${SEED}.log" | \
        grep "final_int8_zlib_roundtrip_exact"

    BPB=$(grep "final_int8_zlib_roundtrip_exact" "atris/logs/seed_${SEED}.log" | \
        grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
    RESULTS+=("$BPB")
    echo "  SEED=$SEED → val_bpb=$BPB"
    echo ""
done

echo "=== RESULTS ==="
echo "Seeds: ${RESULTS[*]}"

# Calculate mean (bash can't do float math, use python)
python3 -c "
import statistics
scores = [float(x) for x in '${RESULTS[*]}'.split()]
mean = statistics.mean(scores)
std = statistics.stdev(scores) if len(scores) > 1 else 0
baseline = 1.2244
improvement = baseline - mean
t_stat = improvement / (std / len(scores)**0.5) if std > 0 else float('inf')
print(f'Mean BPB:      {mean:.8f}')
print(f'Std:           {std:.8f}')
print(f'Improvement:   {improvement:.4f} nats (need >= 0.005)')
print(f't-statistic:   {t_stat:.2f} (need p < 0.01)')
print(f'PASS:          {\"YES\" if improvement >= 0.005 and t_stat > 3.747 else \"NO\"} (t > 3.747 for p<0.01 with df=4)')
"
