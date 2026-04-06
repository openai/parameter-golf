#!/bin/bash
# =============================================================================
# SP4096 ABLATION SUITE (1×H100)
# Phase A: Find best SP4096 base config
# Phase B: Stack architecture techniques
# Phase C: Pre-quant TTT + discriminative TTT
# Updated: 2026-04-05 (fixed env var names)
# =============================================================================
# Run order: A0 (baseline) → A1 (+MuonEq-R) → A2/A3 (arch) → A4 (combo)
#            → B0 (prequant TTT) → B1 (disc TTT) → C0 (GPTQ) → C1 (SLOT)
# Each run: ~5 min wall (300s). No TTT eval (roundtrip only).
# =============================================================================

set -e
cd /workspace/parameter-golf

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TRAIN="train_gpt_full_stack.py"
[ -f "$TRAIN" ] || { echo "ERROR: $TRAIN not found! Run git pull first."; exit 1; }
echo "Using: $TRAIN"
echo ""

# Helper: wait for a log to contain roundtrip result
wait_for_result() {
  local logfile="$1"
  local label="$2"
  echo "Waiting for $label..."
  while ! grep -q "roundtrip" "$logfile" 2>/dev/null && ! grep -q "val_bpb" "$logfile" 2>/dev/null; do
    sleep 30
    if ! tmux list-windows 2>/dev/null | grep -q "$label"; then
      echo "  WARNING: tmux window for $label may have died. Check log: $logfile"
    fi
  done
  echo "  $label complete."
  grep -E "roundtrip|val_bpb|ttt_bpb|slot_bpb" "$logfile" 2>/dev/null | tail -5
}

# Helper: launch a run in a new tmux window
run_ablation() {
  local name="$1"
  shift
  local extra_env="$@"
  local logfile="$LOG_DIR/sp4096_${name}.log"

  if [ -f "$logfile" ] && grep -q "roundtrip\|val_bpb" "$logfile" 2>/dev/null; then
    echo "SKIP (already done): $name"
    return
  fi

  echo "--- Starting: $name ---"
  local cmd="env $extra_env torchrun --nproc_per_node=1 --standalone $TRAIN 2>&1 | tee $logfile; echo 'DONE: $name'"
  tmux new-window -n "${name:0:20}" "bash -c '$cmd'"
}

# =============================================================================
# Common SP4096 base env vars (all runs share these)
# =============================================================================
# Correct env var names verified against train_gpt_full_stack.py:
#   VOCAB_SIZE=4096             → SP4096 vocab
#   TRAIN_SEQ_LEN=4096          → sequence length
#   MLP_MULT=4                  → MLP 4x width
#   WEIGHT_DECAY=0.04           → (note: WEIGHT_DECAY in Muon, not AdamW)
#   MUON_MOMENTUM=0.95          → already default, but explicit is safer
#   WARMDOWN_SCHEDULE=sqrt      → already default in full_stack
#   QK_GAIN_INIT=5.0            → QK-Gain 5.0
#   BIGRAMHASH_DIM=3072         → bigram projection dim
#   MAX_WALLCLOCK_SECONDS=300   → 5-min wall (ablation mode)
#   TTT_ENABLED=0               → NO TTT eval (roundtrip only per policy)
BASE_ENV="
  VOCAB_SIZE=4096
  DATA_PATH=./data/datasets/fineweb10B_sp4096
  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
  TRAIN_SEQ_LEN=4096
  MLP_MULT=4
  MUON_MOMENTUM=0.95
  WARMDOWN_SCHEDULE=sqrt
  QK_GAIN_INIT=5.0
  BIGRAMHASH_DIM=3072
  MAX_WALLCLOCK_SECONDS=300
  TTT_ENABLED=0
"

echo "=== PHASE A: SP4096 BASE CONFIGS ==="
echo "Each run ~5 min. All roundtrip eval only (no TTT)."
echo ""

# A0: Pure SP4096+MLP4x baseline (MuonEq-R OFF to measure its isolated contribution)
run_ablation "A0_sp4096_base" $BASE_ENV MUONEQ_R=0 PARALLEL_START_LAYER=-1 RECUR_LAYERS=""

wait_for_result "$LOG_DIR/sp4096_A0_sp4096_base.log" "A0_sp4096_base"
echo ""

# A1: + MuonEq-R (row normalization before NS, biggest winner: -0.0627)
run_ablation "A1_muoneqr" $BASE_ENV MUONEQ_R=1 PARALLEL_START_LAYER=-1 RECUR_LAYERS=""

wait_for_result "$LOG_DIR/sp4096_A1_muoneqr.log" "A1_muoneqr"
echo ""

# A2: + depth recurrence (3 shared layers: 3,4,5 via RECUR_LAYERS="3,4,5")
# RECUR_LAYERS is a comma-separated list of layer indices to share
run_ablation "A2_depthrecur3" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=-1

# A3: + two-lane parallel residuals (start at layer 4 = PARALLEL_START_LAYER=4)
run_ablation "A3_parallel" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="" PARALLEL_START_LAYER=4

wait_for_result "$LOG_DIR/sp4096_A2_depthrecur3.log" "A2_depthrecur3"
wait_for_result "$LOG_DIR/sp4096_A3_parallel.log" "A3_parallel"
echo ""

# A4: Full arch combo: MuonEq-R + depth recur + parallel resid
run_ablation "A4_full_arch" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4

wait_for_result "$LOG_DIR/sp4096_A4_full_arch.log" "A4_full_arch"
echo ""

echo "=== PHASE B: PRE-QUANT TTT ==="
echo ""

# B0: Best arch + pre-quant AdamW TTT (TTT on FP32 EMA before GPTQ)
# TTT_ENABLED=1 required to activate eval pipeline, TTT_PREQUANT=1 for pre-quant mode
run_ablation "B0_prequant_ttt" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_OPTIMIZER=adamw TTT_LR=0.005 TTT_EPOCHS=6

wait_for_result "$LOG_DIR/sp4096_B0_prequant_ttt.log" "B0_prequant_ttt"
echo ""

# B1: + Discriminative TTT (per-block LR, ULMFiT-style, 10 epochs)
run_ablation "B1_disc_ttt" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_DISCRIMINATIVE=1 TTT_EPOCHS=10

wait_for_result "$LOG_DIR/sp4096_B1_disc_ttt.log" "B1_disc_ttt"
echo ""

echo "=== PHASE C: FULL GPTQ + CAUSAL SLOT ==="
echo ""

# C0: Full Hessian GPTQ (Cholesky + actorder + AR self-gen calibration)
run_ablation "C0_full_gptq" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_DISCRIMINATIVE=1 TTT_EPOCHS=10 \
  GPTQ_FULL_HESSIAN=1 GPTQ_DAMP=0.005

wait_for_result "$LOG_DIR/sp4096_C0_full_gptq.log" "C0_full_gptq"
echo ""

# C1: + Causal SLOT (context-only positions, eval-time only — legality TBD)
run_ablation "C1_causal_slot" $BASE_ENV MUONEQ_R=1 RECUR_LAYERS="3,4,5" PARALLEL_START_LAYER=4 \
  TTT_ENABLED=1 TTT_PREQUANT=1 TTT_DISCRIMINATIVE=1 TTT_EPOCHS=10 \
  GPTQ_FULL_HESSIAN=1 GPTQ_DAMP=0.005 CAUSAL_SLOT_ENABLED=1

wait_for_result "$LOG_DIR/sp4096_C1_causal_slot.log" "C1_causal_slot"
echo ""

echo "==========================="
echo "ALL ABLATIONS COMPLETE!"
echo "==========================="
echo ""
echo "Quick results summary (roundtrip BPB):"
for log in "$LOG_DIR"/sp4096_*.log; do
  name=$(basename "$log" .log)
  bpb=$(grep -E "roundtrip.*bpb|val_bpb" "$log" 2>/dev/null | grep -oP '[0-9]\.[0-9]+' | tail -1)
  echo "  $name: ${bpb:-???}"
done
