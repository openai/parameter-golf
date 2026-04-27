#!/bin/bash
# =============================================================================
# 8×H100 FINAL SUBMISSION SCRIPT
# 3-seed validation run for leaderboard submission
# Updated: 2026-04-05 (uses train_gpt_full_stack.py)
# =============================================================================
# WHEN TO RUN: Only when 1×H100 ablations show our best BPB and fits 16MB.
# COST: ~$21/hr × ~1.5hr total = ~$31-40
# =============================================================================
# ⚠️  NO TTT EVAL unless this is explicitly a submission run.
# ⚠️  NEVER run parallel — all 8 GPUs needed per run.
# =============================================================================
# Usage: bash h100-8x-final-submission.sh [OPTIONAL: config name label]
# Seeds: 4, 30, 2026
# =============================================================================

set -e
cd /workspace/parameter-golf

LABEL="${1:-sp4096_full_stack}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TRAIN="train_gpt_full_stack.py"
[ -f "$TRAIN" ] || { echo "ERROR: $TRAIN not found! Run git pull first."; exit 1; }

echo "========================================"
echo "8×H100 Final Submission Run"
echo "Config: $LABEL"
echo "Script: $TRAIN"
echo "Started: $(date)"
echo "========================================"
echo ""

# =============================================================================
# ⚙️  CONFIGURE BEST CONFIG HERE (update after 1×H100 ablations confirm winner)
# =============================================================================
# SP4096 + MLP4x + MuonEq-R + depth recur + parallel resid + disc TTT + GPTQ
# Adjust based on actual ablation results.
# =============================================================================
BEST_CONFIG="
  VOCAB_SIZE=4096
  DATA_PATH=./data/datasets/fineweb10B_sp4096
  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
  TRAIN_SEQ_LEN=4096
  MLP_MULT=4
  MUON_MOMENTUM=0.95
  MUONEQ_R=1
  WARMDOWN_SCHEDULE=sqrt
  QK_GAIN_INIT=5.0
  BIGRAMHASH_DIM=3072
  RECUR_LAYERS=3,4,5
  PARALLEL_START_LAYER=4
  TTT_ENABLED=1
  TTT_PREQUANT=1
  TTT_DISCRIMINATIVE=1
  TTT_EPOCHS=10
  TTT_LR=0.005
  GPTQ_FULL_HESSIAN=1
  GPTQ_DAMP=0.005
  POLAR_EXPRESS=1
  MAX_WALLCLOCK_SECONDS=590
  GRAD_ACCUM_STEPS=1
"
# Note: GRAD_ACCUM_STEPS=1 on 8×H100 (vs 8 on 1×H100 — torchrun handles total batch)

# =============================================================================
# SEED 4
# =============================================================================
echo "[1/3] Running seed 4..."
env $BEST_CONFIG SEED=4 \
  torchrun --nproc_per_node=8 --standalone "$TRAIN" \
  2>&1 | tee "$LOG_DIR/8x_${LABEL}_seed4.log"
echo ""
echo "Seed 4 complete. BPB:"
grep -E "ttt_bpb|slot_bpb|roundtrip.*bpb|val_bpb" "$LOG_DIR/8x_${LABEL}_seed4.log" | tail -5
echo ""

# =============================================================================
# SEED 30
# =============================================================================
echo "[2/3] Running seed 30..."
env $BEST_CONFIG SEED=30 \
  torchrun --nproc_per_node=8 --standalone "$TRAIN" \
  2>&1 | tee "$LOG_DIR/8x_${LABEL}_seed30.log"
echo ""
echo "Seed 30 complete. BPB:"
grep -E "ttt_bpb|slot_bpb|roundtrip.*bpb|val_bpb" "$LOG_DIR/8x_${LABEL}_seed30.log" | tail -5
echo ""

# =============================================================================
# SEED 2026
# =============================================================================
echo "[3/3] Running seed 2026..."
env $BEST_CONFIG SEED=2026 \
  torchrun --nproc_per_node=8 --standalone "$TRAIN" \
  2>&1 | tee "$LOG_DIR/8x_${LABEL}_seed2026.log"
echo ""
echo "Seed 2026 complete. BPB:"
grep -E "ttt_bpb|slot_bpb|roundtrip.*bpb|val_bpb" "$LOG_DIR/8x_${LABEL}_seed2026.log" | tail -5
echo ""

# =============================================================================
# RESULTS SUMMARY
# =============================================================================
echo "========================================"
echo "3-SEED VALIDATION COMPLETE"
echo "Config: $LABEL"
echo "========================================"
echo ""

# Extract BPB from each seed
extract_bpb() {
  local logfile="$1"
  grep -E "ttt_bpb|discriminative.*bpb|roundtrip.*bpb" "$logfile" 2>/dev/null | \
    grep -oP '[0-9]\.[0-9]{4,}' | tail -1
}

S1=$(extract_bpb "$LOG_DIR/8x_${LABEL}_seed4.log")
S2=$(extract_bpb "$LOG_DIR/8x_${LABEL}_seed30.log")
S3=$(extract_bpb "$LOG_DIR/8x_${LABEL}_seed2026.log")

echo "Seed 4:    ${S1:-???}"
echo "Seed 30:   ${S2:-???}"
echo "Seed 2026: ${S3:-???}"

if command -v python3 &>/dev/null && [ -n "$S1" ] && [ -n "$S2" ] && [ -n "$S3" ]; then
  python3 -c "
s = [$S1, $S2, $S3]
mean = sum(s)/3
std = (sum((x-mean)**2 for x in s)/3)**0.5
print(f'3-seed mean: {mean:.4f} BPB')
print(f'3-seed std:  {std:.4f}')
print(f'Individual:  {s}')
"
fi

echo ""
echo "Artifact sizes:"
ls -lh artifacts/*.lzma artifacts/*.br 2>/dev/null | tail -5 || echo "(no artifacts found in artifacts/)"
echo ""
echo "Logs: $LOG_DIR/8x_${LABEL}_seed*.log"
