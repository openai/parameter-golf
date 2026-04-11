#!/bin/bash
# =============================================================================
# SP4096 ABLATION SUITE (1×H100) — STRICTLY SEQUENTIAL
# Runs one at a time. No parallel runs. No tmux windows per run.
# Updated: 2026-04-05 (fixed: sequential only)
# =============================================================================

set -e
cd /workspace/parameter-golf
mkdir -p logs

TRAIN="train_gpt_full_stack.py"
[ -f "$TRAIN" ] || { echo "ERROR: $TRAIN not found!"; exit 1; }
echo "Using: $TRAIN"
echo "Start: $(date)"
echo ""

# Common SP4096 base env vars
export VOCAB_SIZE=4096
export DATA_PATH=./data/datasets/fineweb10B_sp4096
export TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
export TRAIN_SEQ_LEN=4096
export MLP_MULT=4
export MUON_MOMENTUM=0.95
export WARMDOWN_SCHEDULE=sqrt
export QK_GAIN_INIT=5.0
export BIGRAMHASH_DIM=3072
export MAX_WALLCLOCK_SECONDS=300
export TTT_ENABLED=0

run_sequential() {
  local name="$1"
  local logfile="logs/sp4096_${name}.log"

  if [ -f "$logfile" ] && grep -qE "roundtrip|val_bpb" "$logfile" 2>/dev/null; then
    bpb=$(grep -oP '[0-9]\.[0-9]{4,}' "$logfile" | tail -1)
    echo "SKIP (done): $name  BPB=$bpb"
    return
  fi

  echo ""
  echo "============================================"
  echo "STARTING: $name  ($(date))"
  echo "============================================"

  torchrun --nproc_per_node=1 --standalone "$TRAIN" 2>&1 | tee "$logfile"

  bpb=$(grep -oP '[0-9]\.[0-9]{4,}' "$logfile" | tail -1)
  echo ""
  echo "DONE: $name  BPB=${bpb:-???}  ($(date))"
  echo "============================================"
}

# =============================================================================
# PHASE A: SP4096 base configs
# =============================================================================
echo "=== PHASE A: SP4096 BASE CONFIGS ==="

# A0: Pure baseline (no MuonEq-R)
MUONEQ_R=0 PARALLEL_START_LAYER=-1 RECUR_LAYERS="" \
  run_sequential "A0_sp4096_base"

# A1: + MuonEq-R (biggest winner: -0.0627 on sp1024)
MUONEQ_R=1 PARALLEL_START_LAYER=-1 RECUR_LAYERS="" \
  run_sequential "A1_muoneqr"

# A2: + depth recurrence 3-layer (layers 3,4,5)
MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=-1 \
  run_sequential "A2_depthrecur3"

# A3: + two-lane parallel residuals (start at layer 4)
MUONEQ_R=1 RECUR_LAYERS="" PARALLEL_START_LAYER=4 \
  run_sequential "A3_parallel"

# A4: Full arch combo
MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  run_sequential "A4_full_arch"

# =============================================================================
# PHASE B: Pre-quant TTT
# =============================================================================
echo ""
echo "=== PHASE B: PRE-QUANT TTT ==="

MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_OPTIMIZER=adamw TTT_LR=0.005 TTT_EPOCHS=6 \
  run_sequential "B0_prequant_ttt"

MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_DISCRIMINATIVE=1 TTT_EPOCHS=10 \
  run_sequential "B1_disc_ttt"

# =============================================================================
# PHASE C: Full GPTQ + Causal SLOT
# =============================================================================
echo ""
echo "=== PHASE C: FULL GPTQ + CAUSAL SLOT ==="

MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_DISCRIMINATIVE=1 TTT_EPOCHS=10 \
  GPTQ_FULL_HESSIAN=1 GPTQ_DAMP=0.005 \
  run_sequential "C0_full_gptq"

MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_DISCRIMINATIVE=1 TTT_EPOCHS=10 \
  GPTQ_FULL_HESSIAN=1 GPTQ_DAMP=0.005 CAUSAL_SLOT_ENABLED=1 \
  run_sequential "C1_causal_slot"

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "=== ALL DONE: $(date) ==="
echo ""
echo "Results (roundtrip BPB):"
for log in logs/sp4096_*.log; do
  name=$(basename "$log" .log)
  bpb=$(grep -oP '[0-9]\.[0-9]{4,}' "$log" 2>/dev/null | tail -1)
  echo "  $name: ${bpb:-???}"
done
