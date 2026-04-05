#!/bin/bash
# =============================================================================
# SP4096 ABLATION SUITE
# Phase A: Find best SP4096 base config
# Phase B: Stack architecture techniques  
# Phase C: Pre-quant TTT + discriminative TTT
# Updated: 2026-04-05
# =============================================================================
# All runs use train_gpt_full_stack.py with env vars
# Baseline: 1.49795 BPB (sp1024/512d/MLP2x/XSA4, 15-min wall)
# Our best: 1.2407 BPB (with TTT)
# Target: < 1.10 BPB (Track A SOTA: 1.0807)
# =============================================================================

set -e
cd /workspace/parameter-golf

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

TRAIN="train_gpt_full_stack.py"
# If full_stack not ready yet, fall back to phase0_muoneqr
[ -f "$TRAIN" ] || TRAIN="train_gpt_phase0_muoneqr.py"
echo "Using training script: $TRAIN"

# ---- SP4096 Common env vars ----
SP4096_BASE="
  VOCAB_SIZE=4096
  DATA_PATH=./data/datasets/fineweb10B_sp4096
  TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
  SEQ_LEN=4096
  MLP_MULT=4
  WEIGHT_DECAY=0.085
  MAX_WALLCLOCK_SECONDS=300
"

# ---- Phase A: SP4096 baseline ablations (5-min each) ----
echo ""
echo "=== PHASE A: SP4096 Ablations ==="
echo "Each run ~5 min. Comparing SP4096 base configs."
echo ""

run_ablation() {
  local name="$1"
  local extra_env="$2"
  local logfile="$LOG_DIR/sp4096_${name}.log"

  if [ -f "$logfile" ]; then
    echo "SKIP (already done): $name"
    return
  fi

  echo "--- Starting: $name ---"
  tmux new-window -n "$name" \
    "env $SP4096_BASE $extra_env torchrun --nproc_per_node=1 $TRAIN 2>&1 | tee $logfile; echo 'DONE: $name'"
  sleep 5  # stagger starts to avoid GPU contention
}

# A0: SP4096 baseline (sqrt + MLP4x + WD=0.085) — no special features
run_ablation "A0_sp4096_base" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=0
"

# Wait for A0 before proceeding (it's the reference)
echo "Waiting for A0_sp4096_base to complete..."
while [ ! -f "$LOG_DIR/sp4096_A0_sp4096_base.log" ] || ! grep -q "roundtrip" "$LOG_DIR/sp4096_A0_sp4096_base.log" 2>/dev/null; do
  sleep 30
  echo "  Still waiting for A0..."
done
echo "A0 complete! Launching remaining ablations..."

# A1: + MuonEq-R (the biggest single winner from phase tests, -0.0627)
run_ablation "A1_muoneqr" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
"

# Wait for A1
echo "Waiting for A1_muoneqr..."
while ! grep -q "roundtrip" "$LOG_DIR/sp4096_A1_muoneqr.log" 2>/dev/null; do sleep 30; done

# A2: + depth recurrence 3-layer (layers 3,4,5) — fixed impl, -0.0381 on sp1024
run_ablation "A2_depthrecur3" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  RECUR_LAYERS=3
"

# A3: + two-lane parallel residuals — fixed impl, -0.0321 on sp1024
run_ablation "A3_parallel" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  PARALLEL_RESID=1
"

# Wait for A2+A3
echo "Waiting for A2+A3..."
while ! grep -q "roundtrip" "$LOG_DIR/sp4096_A2_depthrecur3.log" 2>/dev/null; do sleep 30; done
while ! grep -q "roundtrip" "$LOG_DIR/sp4096_A3_parallel.log" 2>/dev/null; do sleep 30; done

# A4: Full combo: MuonEq-R + depth recur + parallel resid
run_ablation "A4_full_arch" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  RECUR_LAYERS=3
  PARALLEL_RESID=1
"

echo "Waiting for A4_full_arch..."
while ! grep -q "roundtrip" "$LOG_DIR/sp4096_A4_full_arch.log" 2>/dev/null; do sleep 30; done

# ---- Phase B: Pre-quant TTT on best config ----
echo ""
echo "=== PHASE B: Pre-quant TTT ==="
echo ""

# B0: Best Phase A config + pre-quant TTT (pipeline fix, -0.0314 on sp1024)
run_ablation "B0_prequant_ttt" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  RECUR_LAYERS=3
  PARALLEL_RESID=1
  TTT_PREQUANT=1
  TTT_LR=0.005
  TTT_EPOCHS=6
"

# B1: + Discriminative TTT (per-block LR, -0.0276 on sp1024, SOTA technique)
run_ablation "B1_disc_ttt" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  RECUR_LAYERS=3
  PARALLEL_RESID=1
  TTT_PREQUANT=1
  TTT_DISCRIMINATIVE=1
  TTT_EPOCHS=10
"

echo "Waiting for B0+B1..."
while ! grep -q "roundtrip" "$LOG_DIR/sp4096_B0_prequant_ttt.log" 2>/dev/null; do sleep 30; done
while ! grep -q "roundtrip" "$LOG_DIR/sp4096_B1_disc_ttt.log" 2>/dev/null; do sleep 30; done

# ---- Phase C: Full GPTQ + causal SLOT ----
echo ""
echo "=== PHASE C: Full GPTQ + Causal SLOT ==="
echo ""

# C0: + Full Hessian GPTQ (replaces GPTQ-lite, -0.0198 on sp1024)
run_ablation "C0_full_gptq" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  RECUR_LAYERS=3
  PARALLEL_RESID=1
  TTT_PREQUANT=1
  TTT_DISCRIMINATIVE=1
  TTT_EPOCHS=10
  GPTQ_FULL_HESSIAN=1
"

# C1: + Causal SLOT (eval-time only, -0.0284 on sp1024 — legality TBD)
run_ablation "C1_causal_slot" "
  MUON_MOMENTUM=0.95
  SQRT_WARMDOWN=1
  QK_GAIN=5.0
  BIGRAM_DIM=3072
  MUONEQR=1
  RECUR_LAYERS=3
  PARALLEL_RESID=1
  TTT_PREQUANT=1
  TTT_DISCRIMINATIVE=1
  TTT_EPOCHS=10
  GPTQ_FULL_HESSIAN=1
  CAUSAL_SLOT_ENABLED=1
"

echo "All ablations launched. Check logs in $LOG_DIR/"
echo ""
echo "Quick results summary:"
grep -h "roundtrip\|ttt_bpb\|slot_bpb" "$LOG_DIR"/sp4096_*.log 2>/dev/null | sort || echo "No results yet."
