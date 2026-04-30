#!/usr/bin/env bash
# Reproduce PR #1493 SOTA seed=42. This is Experiment 001.
#
# Run on a 1×H100 pod (or up — set NPROC accordingly).
# Expected outcome: val_bpb_ttt ≈ 1.08079 ± 0.005

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

NPROC="${NPROC:-1}"
SEED="${SEED:-42}"

mkdir -p Opus/experiments/logs Opus/artifacts

LOG="Opus/experiments/logs/001_repro_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"

echo "Running SOTA reproduction → $LOG"
echo "  NPROC=$NPROC  SEED=$SEED"
echo "  artifact: records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py"
echo

SEED=$SEED \
    TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
    QK_GAIN_INIT=5.25 \
    RUN_ID="opus_e001_seed${SEED}" \
    torchrun --standalone --nproc_per_node="$NPROC" \
        records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py \
        2>&1 | tee "$LOG"

echo
echo "Final val_bpb lines from log:"
grep -E "(val_bpb|val_loss).*ttt|quantized" "$LOG" | tail -10

# Save the artifact for Day 2 reuse
if [ -f final_model.int6.ptz ]; then
    cp final_model.int6.ptz "Opus/artifacts/seed${SEED}.int6.ptz"
    echo "Saved checkpoint: Opus/artifacts/seed${SEED}.int6.ptz ($(du -h final_model.int6.ptz | cut -f1))"
fi
