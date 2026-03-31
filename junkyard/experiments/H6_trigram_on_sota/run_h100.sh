#!/bin/bash
# H6: Trigram vs Bigram on SOTA f1 base
# Two arms at 0.25 scale (150s wallclock), single GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt_h6.py"

# Common config: 0.25 scale (150s wallclock), single GPU
export NPROC=1
export MAX_WALLCLOCK_SECONDS=150
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100

echo "========================================="
echo "H6 Arm A: NGRAM_TYPE=bigram (control)"
echo "========================================="
NGRAM_TYPE=bigram RUN_ID="h6_arm_a_bigram_$(date +%Y%m%d_%H%M%S)" \
    python "$TRAIN_SCRIPT" 2>&1 | tee "${SCRIPT_DIR}/results/arm_a_bigram.log"

echo ""
echo "========================================="
echo "H6 Arm B: NGRAM_TYPE=trigram"
echo "========================================="
NGRAM_TYPE=trigram RUN_ID="h6_arm_b_trigram_$(date +%Y%m%d_%H%M%S)" \
    python "$TRAIN_SCRIPT" 2>&1 | tee "${SCRIPT_DIR}/results/arm_b_trigram.log"

echo ""
echo "========================================="
echo "H6 complete — compare final bpb in logs"
echo "========================================="
grep -h "final_int6_sliding_window_exact\|legal_ttt_exact\|ngram_type" \
    "${SCRIPT_DIR}/results/arm_a_bigram.log" \
    "${SCRIPT_DIR}/results/arm_b_trigram.log" || true
