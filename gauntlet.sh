#!/usr/bin/env bash
# ============================================================
# GAUNTLET TEST — parameter-golf experiment runner
# Runs all experiments sequentially, records BPB for each.
# Designed for 1×H100 ablation mode (cheap, ~$3.50/hr)
#
# Usage:
#   bash gauntlet.sh [--steps N] [--vocab V] [--gpus N]
#
# Options:
#   --steps N     Training iterations per experiment  (default: 2000)
#   --vocab V     Vocabulary size: 1024 or 8192       (default: 1024)
#   --gpus N      Number of GPUs per run              (default: 1)
#   --deq-only    Only run DEQ experiment
#   --seeds-only  Only run Seed-LoRA experiment
#   --incr-only   Only run incremental submission
#   --skip-deq    Skip DEQ (it's slower per step)
#
# Output: gauntlet_results.txt with timestamped BPB for every run
# ============================================================
# NOTE: do not use set -e — experiments are allowed to fail individually

STEPS=2000
VOCAB=1024
GPUS=1
SKIP_DEQ=0
SKIP_BASELINE=0
INCR_ONLY=0
DEQ_ONLY=0
SEEDS_ONLY=0
MOD_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)   STEPS="$2";    shift 2 ;;
    --vocab)   VOCAB="$2";    shift 2 ;;
    --gpus)    GPUS="$2";     shift 2 ;;
    --skip-deq) SKIP_DEQ=1;   shift ;;
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    --incr-only) INCR_ONLY=1; shift ;;
    --deq-only)  DEQ_ONLY=1;  shift ;;
    --seeds-only) SEEDS_ONLY=1; shift ;;
    --mod-only)  MOD_ONLY=1;  shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---- paths ----
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$REPO_DIR/data/datasets/fineweb10B_sp${VOCAB}"
TOK_DIR="$REPO_DIR/data/tokenizers"
TOKENIZER_PATH="$TOK_DIR/fineweb_${VOCAB}_bpe.model"
RESULTS="$REPO_DIR/gauntlet_results.txt"
LOG_DIR="$REPO_DIR/logs/gauntlet"
mkdir -p "$LOG_DIR"

echo "========================================"
echo " GAUNTLET TEST — $(date)"
echo " steps=$STEPS  vocab=sp${VOCAB}  gpus=$GPUS"
echo "========================================"
echo ""

# ---- check data ----
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data not found at $DATA_DIR"
  echo "Run: python3 data/cached_challenge_fineweb.py --variant sp${VOCAB} --train-shards 5"
  echo "Or check: tail -f /tmp/data_download.log"
  exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
  echo "ERROR: Tokenizer not found at $TOKENIZER_PATH"
  exit 1
fi

echo "Data: $DATA_DIR  ✓"
echo "Tokenizer: $TOKENIZER_PATH  ✓"
echo ""

# ---- helper ----
run_experiment() {
  local NAME="$1"
  local SCRIPT="$2"
  local EXTRA_ENV="${3:-}"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ▶ $NAME"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [ ! -f "$SCRIPT" ]; then
    echo "  SKIP: script not found: $SCRIPT"
    return
  fi

  RUN_ID="gauntlet_${NAME}_$(date +%H%M%S)"
  LOGFILE="$LOG_DIR/${RUN_ID}.log"
  local T_START
  T_START=$(date +%s)

  # On 1 GPU, don't accumulate 8 microbatches — use the full batch in one pass
  ACCUM_STEPS=$(( 8 / GPUS ))
  [[ $ACCUM_STEPS -lt 1 ]] && ACCUM_STEPS=1

  env \
    RUN_ID="$RUN_ID" \
    VOCAB_SIZE="$VOCAB" \
    DATA_DIR="$REPO_DIR/data" \
    ITERATIONS="$STEPS" \
    VAL_LOSS_EVERY=500 \
    MAX_WALLCLOCK_SECONDS=580 \
    SLIDING_WINDOW_ENABLED=1 \
    TRAIN_BATCH_TOKENS=786432 \
    $EXTRA_ENV \
  torchrun --standalone --nproc_per_node="$GPUS" "$SCRIPT" 2>&1 | tee "$LOGFILE"

  local T_END
  T_END=$(date +%s)
  local ELAPSED=$(( T_END - T_START ))

  # Extract best val_bpb from log
  local BEST_BPB
  BEST_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$LOGFILE" | sort -n | head -1 || echo "N/A")
  local LAST_BPB
  LAST_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$LOGFILE" | tail -1 || echo "N/A")

  echo ""
  echo "  ✓ DONE — elapsed: ${ELAPSED}s | best_bpb: $BEST_BPB | last_bpb: $LAST_BPB"

  # Append to results file
  echo "$(date '+%Y-%m-%d %H:%M:%S') | experiment=${NAME} | steps=${STEPS} | vocab=sp${VOCAB} | gpus=${GPUS} | best_bpb=${BEST_BPB} | last_bpb=${LAST_BPB} | elapsed=${ELAPSED}s | log=${LOGFILE}" \
    >> "$RESULTS"
}

# ======================================================
# 0. BASELINE — original train_gpt.py (control group)
# ======================================================
if [[ $INCR_ONLY -eq 0 && $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $MOD_ONLY -eq 0 && $SKIP_BASELINE -eq 0 ]]; then
  run_experiment "baseline" \
    "$REPO_DIR/train_gpt.py"
fi

# ======================================================
# 1. INCREMENTAL SUBMISSION
#    QK-Gain 5.5 + 4-loop recurrence + early parallel residuals + selective TTT
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $MOD_ONLY -eq 0 ]]; then
  INCR_SCRIPT=$(ls -t "$REPO_DIR"/records/track_10min_16mb/2026-04-23_QK55_*/train_gpt.py 2>/dev/null | head -1)
  if [ -n "$INCR_SCRIPT" ]; then
    run_experiment "incr_QK55_4loop" "$INCR_SCRIPT"
  else
    echo "WARN: incremental submission not found, skipping"
  fi
fi

# ======================================================
# 2. ABLATION: QK-Gain 5.5 alone (no loop change)
#    Lets us isolate the gain of QK 5.5 vs 5.25
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 && $MOD_ONLY -eq 0 ]]; then
  run_experiment "ablation_QK55_only" \
    "$REPO_DIR/train_gpt.py" \
    "QK_GAIN_INIT=5.5"
fi

# ======================================================
# 3. ABLATION: 4-loop only (QK stays at 5.25)
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 && $MOD_ONLY -eq 0 ]]; then
  run_experiment "ablation_4loop_only" \
    "$REPO_DIR/train_gpt.py" \
    "NUM_LOOPS=3"
fi

# ======================================================
# 4. DEQ UNIVERSAL TRANSFORMER
#    1 physical block → fixed-point convergence (Anderson acceleration)
# ======================================================
if [[ $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 && $SKIP_DEQ -eq 0 && $MOD_ONLY -eq 0 ]]; then
  run_experiment "deq_universal" \
    "$REPO_DIR/experiments/train_gpt_deq.py" \
    "DEQ_MAX_ITER_TRAIN=8 DEQ_PHANTOM_STEPS=4 DEQ_MAX_ITER_EVAL=16 DEQ_TOL=1e-3"
fi

# ======================================================
# 5. SEED-LORA
#    Random basis weights (seeded) + rank-8/4 LoRA adapters only stored
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $INCR_ONLY -eq 0 && $MOD_ONLY -eq 0 ]]; then
  run_experiment "seed_lora_r8" \
    "$REPO_DIR/experiments/train_gpt_seeds.py" \
    "LORA_RANK_ATTN=8 LORA_RANK_MLP=4 LEARN_RANDOM_SCALE=1"
fi

# ======================================================
# 6. SEED-LORA HIGH RANK
#    Since we have budget left, try rank-32 (still tiny vs 16MB)
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $INCR_ONLY -eq 0 && $MOD_ONLY -eq 0 ]]; then
  run_experiment "seed_lora_r32" \
    "$REPO_DIR/experiments/train_gpt_seeds.py" \
    "LORA_RANK_ATTN=32 LORA_RANK_MLP=16 LEARN_RANDOM_SCALE=1"
fi

# ======================================================
# 7. MIXTURE OF DEPTHS — 50% routing capacity
#    ~2× faster training → more steps in 10 min → lower BPB
#    On OpenAI's wish list. Explicitly mentioned in README.
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 ]]; then
  run_experiment "mod_capacity50" \
    "$REPO_DIR/experiments/train_gpt_mod.py" \
    "MOD_CAPACITY=0.5 MOD_LAYERS=all MOD_AUX_LOSS_COEF=0.01"
fi

# ======================================================
# 8. MIXTURE OF DEPTHS — 25% routing (more aggressive)
#    Skips 75% of tokens per layer — even faster training
#    but may hurt quality. Tests the tradeoff.
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 ]]; then
  run_experiment "mod_capacity25" \
    "$REPO_DIR/experiments/train_gpt_mod.py" \
    "MOD_CAPACITY=0.25 MOD_LAYERS=all MOD_AUX_LOSS_COEF=0.005"
fi

# ======================================================
# 9. SPECULATIVE MUON — faster Newton-Schulz (2 steps vs 5)
#    ~40% less optimizer compute → more training steps in 10 min
#    Free ablation: just change MUON_BACKEND_STEPS
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 ]]; then
  INCR_SCRIPT=$(ls -t "$REPO_DIR"/records/track_10min_16mb/2026-04-23_QK55_*/train_gpt.py 2>/dev/null | head -1)
  if [ -n "$INCR_SCRIPT" ]; then
    run_experiment "speculative_muon_ns2" \
      "$INCR_SCRIPT" \
      "MUON_BACKEND_STEPS=2"
  fi
fi

# ======================================================
# 10. SPECULATIVE MUON — 3 steps (middle ground)
# ======================================================
if [[ $DEQ_ONLY -eq 0 && $SEEDS_ONLY -eq 0 && $INCR_ONLY -eq 0 ]]; then
  INCR_SCRIPT=$(ls -t "$REPO_DIR"/records/track_10min_16mb/2026-04-23_QK55_*/train_gpt.py 2>/dev/null | head -1)
  if [ -n "$INCR_SCRIPT" ]; then
    run_experiment "speculative_muon_ns3" \
      "$INCR_SCRIPT" \
      "MUON_BACKEND_STEPS=3"
  fi
fi

# ======================================================
# RESULTS SUMMARY
# ======================================================
echo ""
echo "════════════════════════════════════════════════"
echo "  GAUNTLET COMPLETE — $(date)"
echo "════════════════════════════════════════════════"
echo ""
if [ -f "$RESULTS" ]; then
  echo "Results (sorted by best_bpb):"
  echo ""
  # Print header
  printf "  %-30s  %-10s  %-10s  %-8s\n" "EXPERIMENT" "BEST_BPB" "LAST_BPB" "TIME"
  printf "  %-30s  %-10s  %-10s  %-8s\n" "----------" "--------" "--------" "----"
  # Parse and sort results
  grep "$(date '+%Y-%m-%d')" "$RESULTS" 2>/dev/null | \
  awk -F'|' '
  {
    for(i=1;i<=NF;i++){
      if($i ~ /experiment=/) { split($i,a,"="); name=a[2] }
      if($i ~ /best_bpb=/)   { split($i,a,"="); best=a[2] }
      if($i ~ /last_bpb=/)   { split($i,a,"="); last=a[2] }
      if($i ~ /elapsed=/)    { split($i,a,"="); elapsed=a[2] }
    }
    print best, name, last, elapsed
  }' | sort -n | \
  while read bpb name last_bpb elapsed; do
    printf "  %-30s  %-10s  %-10s  %-8s\n" "$name" "$bpb" "$last_bpb" "$elapsed"
  done
  echo ""
  echo "Current SOTA: 1.0810 bpb (bigbag, 2026-04-09)"
fi
echo ""
echo "Full results: $RESULTS"
echo "Logs: $LOG_DIR/"
