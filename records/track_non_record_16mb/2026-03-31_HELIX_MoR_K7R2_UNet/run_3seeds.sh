#!/bin/bash
# Run HELIX MoR K7R2 UNet across 3 seeds for statistical submission proof.
# Each seed gets its own directory with logs, final_model.pt, final_model.int6.ptz.
#
# Usage (from /workspace/parameter-golf):
#   bash records/track_10min_16mb/2026-03-31_HELIX_MoR_K7R2_UNet/run_3seeds.sh
#
# Output layout:
#   /workspace/runs/helix_mor_k7r2_unet/
#     seed_1337/
#       logs/helix_mor_k7r2_unet_seed1337.txt
#       final_model.pt
#       final_model.int6.ptz
#     seed_1338/  (same)
#     seed_1339/  (same)

set -e

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$REPO/records/track_10min_16mb/2026-03-31_HELIX_MoR_K7R2_UNet/train_gpt.py"
DATA_PATH="$REPO/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="$REPO/data/tokenizers/fineweb_1024_bpe.model"
RUNS_DIR="/workspace/runs/helix_mor_k7r2_unet"
SEEDS=(1337 1338 1339)

echo "Repo:   $REPO"
echo "Script: $SCRIPT"
echo "Data:   $DATA_PATH"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo " Starting seed=$SEED"
    echo "=========================================="

    RUN_DIR="$RUNS_DIR/seed_${SEED}"
    mkdir -p "$RUN_DIR"
    cd "$RUN_DIR"

    RUN_ID="helix_mor_k7r2_unet_seed${SEED}" \
    SEED="$SEED" \
    NUM_UNIQUE_BLOCKS=7 \
    NUM_ITERATIONS=2 \
    WARMDOWN_ITERS=2500 \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    VOCAB_SIZE=1024 \
    TTT_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 \
        "$SCRIPT" \
        2>&1 | tee "seed_${SEED}_stdout.log"

    echo ""
    echo "--- seed=$SEED result ---"
    grep -E "legal_ttt_exact|final_int6_sliding_window_s64_exact|Serialized model int6" \
        "seed_${SEED}_stdout.log" | tail -4
    echo ""

    cd "$REPO"
done

echo "=========================================="
echo " ALL 3 SEEDS COMPLETE"
echo "=========================================="
echo ""
echo "BPB Summary:"
for SEED in "${SEEDS[@]}"; do
    BPB=$(grep "legal_ttt_exact" "$RUNS_DIR/seed_${SEED}/seed_${SEED}_stdout.log" \
          | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    echo "  seed=$SEED  legal_ttt bpb=$BPB"
done

echo ""
echo "Artifacts:"
for SEED in "${SEEDS[@]}"; do
    DIR="$RUNS_DIR/seed_${SEED}"
    echo "  seed=$SEED:"
    echo "    log:   $DIR/seed_${SEED}_stdout.log"
    echo "    model: $DIR/final_model.int6.ptz"
done
