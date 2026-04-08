#!/bin/bash
# ============================================================================
# INSIDE-OUT TRAINING ABLATION — 5 MIN × 4 CONFIGS
#
# Question: Does progressive layer unfreezing from bottleneck outward help?
# Which phase-1 variant is better?
#
# Configs:
#   A — Baseline       : all layers always active, no inside-out
#   B — IO-capsule     : phase1=capsule only (all TF layers frozen)
#   C — IO-inner       : phase1 near-zero → jump straight to progressive unfreeze
#   D — IO-coordinated : IO phases aligned with curriculum (5%/20%)
#
# Fixed: 8L dim=256, curriculum 64→256→1024 @5%/20%, XSA=999, seed=42
# Fine-grained logging (every 10 steps) to study loss curve shape
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
mkdir -p logs
RESULTS="/tmp/ablation_io_results_${TS}.txt"

run_config() {
    local NAME=$1
    shift
    local LOG="logs/ablation_io_${NAME}_${TS}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUNNING CONFIG: $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    RUN_ID="ablation_io_${NAME}_${TS}" \
    ARCHITECTURE=skc \
    NUM_LAYERS=8  MODEL_DIM=256  NUM_HEADS=4  NUM_KV_HEADS=2  MLP_MULT=4 \
    VOCAB_SIZE=1024 \
    SKC_BLOCK_SIZE=16  SKC_NUM_CAPSULES=16  SKC_CAPSULE_DIM=64  SKC_CONV_KERNEL=4 \
    XSA_START_LAYER=999 \
    BIGRAM_HASH_ENABLED=1  BIGRAM_HASH_BUCKETS=3072  BIGRAM_HASH_DIM=112 \
    ENGRAM_NUM_HEADS=4  ENGRAM_NUM_ORDERS=3  ENGRAM_INJECT_LAYER=1 \
    PARTIAL_ROPE_DIMS=16  LN_SCALE_DAMPING=1 \
    TRAIN_SEQ_LEN=1024  TRAIN_BATCH_TOKENS=16384  GRAD_ACCUM_STEPS=4 \
    MLX_MAX_MICROBATCH_TOKENS=8192  MLX_EAGER_EVAL=1 \
    MAX_WALLCLOCK_SECONDS=300  ITERATIONS=1000000 \
    WARMUP_STEPS=5  WARMDOWN_FRACTION=0.3 \
    CURRICULUM_ENABLED=1 \
    CURRICULUM_PHASE1_SEQ=64   CURRICULUM_PHASE2_SEQ=256 \
    CURRICULUM_PHASE1_FRAC=0.05  CURRICULUM_PHASE2_FRAC=0.20 \
    MATRIX_LR=0.02  SCALAR_LR=0.015  TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.95  MUON_MOMENTUM_WARMUP_STEPS=0  MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04  ADAM_WD=0.04  GRAD_CLIP_NORM=0.3 \
    LAWA_ENABLED=1  LAWA_K=5  LAWA_FREQ=100 \
    SWA_ENABLED=1   SWA_EVERY=50  SMEARGATE_ENABLED=1  TKO_ENABLED=0 \
    FEEDBACK_ENABLED=0  CAPSULE_ENABLED=0  VRL_ENABLED=0 \
    TTT_ENABLED=0  EMA_ENABLED=0  MOE_ENABLED=0 \
    GPTQ_LITE_ENABLED=1  TURBO_QUANT_EXPORT=1  TURBO_QUANT_TRAIN=0  TURBO_QUANT_KV=1 \
    NGRAM_CACHE_ENABLED=1  NGRAM_MAX_ORDER=5 \
    NGRAM_ALPHA_BASE=0.05  NGRAM_ALPHA_SCALE=0.55  NGRAM_ENTROPY_CENTER=4.0 \
    SLIDING_EVAL=1  SLIDING_EVAL_STRIDE=64  TEMP_SCALING=1 \
    TRAIN_LOG_EVERY=10  VAL_BATCH_SIZE=65536  VAL_LOSS_EVERY=0 \
    SEED=42 \
    env "$@" \
    bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

    # Parse results
    STEPS=$(grep "^step:" "$LOG" | grep -v "val_loss" | tail -1 | sed 's/step:\([0-9]*\)\/.*/\1/')
    BPB_N=$(grep "ngram_cache" "$LOG" | grep -o 'val_bpb:[0-9.]*' | tail -1 | cut -d: -f2)
    BPB_F=$(grep "final_eval"  "$LOG" | grep -o 'val_bpb:[0-9.]*' | tail -1 | cut -d: -f2)
    LOSS_F=$(grep "^step:" "$LOG" | grep -v "val_loss" | tail -1 | grep -o 'loss:[0-9.]*' | cut -d: -f2)
    echo "${NAME}|${STEPS:-?}|${LOSS_F:-?}|${BPB_F:-?}|${BPB_N:-?}" >> "$RESULTS"
    echo "  → steps=${STEPS:-?}  loss=${LOSS_F:-?}  bpb_final=${BPB_F:-?}  bpb_ngram=${BPB_N:-?}"
}

echo "════════════════════════════════════════════════════════"
echo "  INSIDE-OUT ABLATION — 5min × 4 configs"
echo "  8L dim=256 | curriculum 64→256→1024 @5%/20% | seed=42"
echo "  Watching: loss curve shape + final BPB"
echo "════════════════════════════════════════════════════════"

# ── Config A: Baseline ────────────────────────────────────────────────────────
# No inside-out. All 8 layers always active from step 0.
# This is the control.
run_config "A_baseline" \
    INSIDE_OUT_TRAINING=0

# ── Config B: IO-capsule ──────────────────────────────────────────────────────
# Phase 1 (0→20%): capsule+Koopman only, ALL transformer layers frozen
# Phase 2 (20→55%): progressive unfreeze from bottleneck outward
#   Step 0-60:   dist=-1 (frozen)  → only capsule/Koopman trains
#   Step 60-165: dist increases → 3,4 → 2,5 → 1,6 → 0,7
#   Step 165+:   all open
run_config "B_io_capsule" \
    INSIDE_OUT_TRAINING=1  INSIDE_OUT_PHASE1_FRAC=0.20  INSIDE_OUT_PHASE2_FRAC=0.55

# ── Config C: IO-inner ────────────────────────────────────────────────────────
# Skips the all-frozen phase. Phase1≈0 → jumps immediately into progressive
# unfreezing. Inner layers (3,4 dist=0) train first, then 2,5, then 1,6, then 0,7.
# Tests: is the bottleneck-first ordering itself valuable (without capsule-only warmup)?
run_config "C_io_inner" \
    INSIDE_OUT_TRAINING=1  INSIDE_OUT_PHASE1_FRAC=0.01  INSIDE_OUT_PHASE2_FRAC=0.45

# ── Config D: IO-coordinated ──────────────────────────────────────────────────
# Phases locked to curriculum transitions:
#   seq=64  (0→5%):  phase1 — capsule+Koopman only (tiny fast steps, capsule anchors)
#   seq=256 (5→20%): phase2 — progressive unfreeze (inner layers see medium context)
#   seq=1024 (20%+): all layers open + warmdown
# Hypothesis: curriculum phase 1 is the right time to anchor the capsule bank.
#   When seq=64 the outer layers have no long-range signal anyway —
#   freezing them costs nothing and lets the bottleneck anchor cleanly.
run_config "D_io_coordinated" \
    INSIDE_OUT_TRAINING=1  INSIDE_OUT_PHASE1_FRAC=0.05  INSIDE_OUT_PHASE2_FRAC=0.20

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════════"
python3 - <<'PYEOF'
import os, sys
f = '/tmp/ablation_io_results_' + os.environ.get('TS', '') + '.txt'
# fallback: find by glob
if not os.path.exists(f):
    import glob
    files = sorted(glob.glob('/tmp/ablation_io_results_*.txt'))
    if files: f = files[-1]

labels = {
    'A_baseline':       'A  baseline       all layers always on              ',
    'B_io_capsule':     'B  io-capsule     phase1=capsule only (0→20%)       ',
    'C_io_inner':       'C  io-inner       phase1≈skip→inner first (0→45%)   ',
    'D_io_coordinated': 'D  io-coordinated phases match curriculum (5%/20%)  ',
}

rows = []
for line in open(f):
    parts = line.strip().split('|')
    if len(parts) == 5:
        name, steps, loss, bpb_f, bpb_n = parts
        try: score = float(bpb_n)
        except: score = 99.0
        rows.append((score, labels.get(name, name), steps, loss, bpb_f, bpb_n))
rows.sort()

print(f"  {'Config':<50} {'steps':>6}  {'loss':>6}  {'bpb_f':>7}  {'bpb_n':>7}")
print(f"  {'─'*50} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
for i, (score, label, steps, loss, bpb_f, bpb_n) in enumerate(rows):
    star = ' ← BEST' if i == 0 else ''
    print(f"  {label} {steps:>6}  {loss:>6}  {bpb_f:>7}  {bpb_n:>7}{star}")

print()
if len(rows) >= 2:
    best_n  = rows[0][5]
    base_n  = next((r[5] for r in rows if 'baseline' in r[1]), None)
    if base_n and best_n != '?':
        delta = float(best_n) - float(base_n)
        print(f"  Delta best vs baseline: {delta:+.4f} BPB")
PYEOF
echo "  Full logs: logs/ablation_io_*_${TS}.log"
