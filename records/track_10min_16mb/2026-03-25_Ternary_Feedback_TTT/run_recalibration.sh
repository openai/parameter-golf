#!/bin/bash
# ============================================================================
# RECALIBRATION — Pre-H100 config validation
# Architecture: SKC 8L dim=256 vocab=1024 (proven best local config)
#
# Two runs, 20 minutes total:
#
#  FA — CURRENT BEST (exact replica of 1.6552 BPB baseline)
#       XSA=off, BigramHash=4096×128, engram_orders=3
#       → Confirms reproducibility + clean comparison anchor
#
#  FB — H100 CANDIDATE (winner's config choices on our model)
#       XSA=ALL_LAYERS, BigramHash=3072×112, engram_orders=3
#       → If this beats FA, use XSA-all + 3072×112 on H100
#       → If it doesn't, keep our proven config unchanged
#
# After both runs: pick the better BPB for H100.
#
# Usage:
#   bash run_recalibration.sh
# ============================================================================
set -uo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR" || exit 1

PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/bin/python3}
[[ ! -f "$PYTHON_BIN" ]] && PYTHON_BIN=python3

TS=$(date +%s)
MASTER_LOG="${EXPDIR}/recalibration_${TS}.log"

log() { echo "$*" | tee -a "$MASTER_LOG"; }

run_exp() {
    local NAME=$1; shift
    local LOGFILE="${EXPDIR}/recal_${NAME}_${TS}.log"

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  RUN ${NAME}  $(date)"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Proven base (8L dim=256, identical to 1.6552 BPB run) ──────────────────
    env \
    ARCHITECTURE=skc \
    NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    VOCAB_SIZE=1024 \
    \
    TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=131072 GRAD_ACCUM_STEPS=4 \
    MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 \
    MAX_WALLCLOCK_SECONDS=600 ITERATIONS=100000 \
    WARMUP_STEPS=5 WARMDOWN_FRACTION=0.5 \
    \
    SKC_BLOCK_SIZE=16 SKC_NUM_CAPSULES=16 SKC_CAPSULE_DIM=64 SKC_CONV_KERNEL=4 \
    \
    FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 VRL_ENABLED=0 TTT_ENABLED=0 \
    EMA_ENABLED=0 MOE_ENABLED=0 TKO_ENABLED=0 \
    \
    BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3 ENGRAM_NUM_HEADS=4 \
    ENGRAM_INJECT_LAYER=1 \
    \
    LN_SCALE_DAMPING=1 PARTIAL_ROPE_DIMS=16 \
    \
    LAWA_ENABLED=1 LAWA_K=5 LAWA_FREQ=100 \
    SWA_ENABLED=1 SWA_EVERY=50 SMEARGATE_ENABLED=1 \
    \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=256 \
    CURRICULUM_PHASE1_FRAC=0.05 CURRICULUM_PHASE2_FRAC=0.20 \
    STOCHASTIC_DEPTH_PROB=0 \
    \
    MATRIX_LR=0.02 SCALAR_LR=0.015 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_STEPS=0 MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
    \
    GPTQ_LITE_ENABLED=1 TURBO_QUANT_EXPORT=1 TURBO_QUANT_TRAIN=0 TURBO_QUANT_KV=1 \
    NGRAM_CACHE_ENABLED=1 NGRAM_MAX_ORDER=5 \
    NGRAM_ALPHA_BASE=0.05 NGRAM_ALPHA_SCALE=0.55 NGRAM_ENTROPY_CENTER=4.0 \
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=64 TEMP_SCALING=1 \
    TRAIN_LOG_EVERY=200 VAL_BATCH_SIZE=65536 \
    SEED=42 \
    \
    "$@" bash run_mlx_reasoner.sh 2>&1 | tee "$LOGFILE"

    # ── Parse result ────────────────────────────────────────────────────────────
    local BPB STEPS SIZE_MB
    BPB=$("$PYTHON_BIN" -c "
import re
with open('${LOGFILE}') as f: c = f.read()
# Prefer ngram-boosted BPB, then sliding, then plain val
for pat in [r'ngram.*val_bpb:([\d.]+)', r'sliding.*val_bpb:([\d.]+)', r'val_bpb:([\d.]+)']:
    m = re.findall(pat, c)
    if m: print(float(m[-1])); break
else: print('N/A')
" 2>/dev/null)

    STEPS=$("$PYTHON_BIN" -c "
import re
with open('${LOGFILE}') as f: c = f.read()
m = re.findall(r'step:(\d+)/', c)
print(m[-1] if m else 'N/A')
" 2>/dev/null)

    SIZE_MB=$("$PYTHON_BIN" -c "
import re
with open('${LOGFILE}') as f: c = f.read()
m = re.findall(r'artifact_size_bytes:(\d+)|(\d+)\s*/\s*16000000', c)
if m:
    val = next(v for v in m[-1] if v)
    print(f'{int(val)/1024/1024:.2f}MB')
else: print('N/A')
" 2>/dev/null)

    log "  ✓ ${NAME}: bpb=${BPB}  steps=${STEPS}  size=${SIZE_MB}"
    echo "RESULT ${NAME}: bpb=${BPB} steps=${STEPS} size=${SIZE_MB}" >> "$MASTER_LOG"
}

log "========================================================"
log "  PARAMETER GOLF — PRE-H100 RECALIBRATION"
log "  $(date)"
log "  Two 10-min runs to lock in H100 config"
log "========================================================"

# ── RUN FA: CURRENT BEST (exact 1.6552 replica) ──────────────────────────────
# XSA disabled (our proven best had XSA_START_LAYER=999)
# BigramHash 4096×128 (our proven best)
log ""
log ">>> FA: Current best replica (XSA=off, bigram=4096×128)"
log "    Expected: ~1.655 BPB  [sanity check]"
run_exp "FA_current_best" \
    XSA_START_LAYER=999 \
    BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128

# ── RUN FB: H100 CANDIDATE (winner's choices on our model) ───────────────────
# XSA on ALL layers (winner's novel contribution — never tested on our SKC)
# BigramHash 3072×112 (winner's setting — slightly less params, more compressed)
log ""
log ">>> FB: H100 candidate (XSA=ALL_LAYERS, bigram=3072×112)"
log "    Hypothesis: XSA-all helps even with SKC spectral mixing"
run_exp "FB_h100_candidate" \
    XSA_START_LAYER=0 \
    BIGRAM_HASH_BUCKETS=3072 BIGRAM_HASH_DIM=112

# ── Final verdict ─────────────────────────────────────────────────────────────
log ""
log "========================================================"
log "  RECALIBRATION COMPLETE"
log "========================================================"
log ""
log "Results:"
grep "^RESULT" "$MASTER_LOG"
log ""

FA_BPB=$("$PYTHON_BIN" -c "
import re
with open('${MASTER_LOG}') as f: c = f.read()
m = re.search(r'RESULT FA.*bpb=([\d.]+)', c)
print(float(m.group(1)) if m else 999)
" 2>/dev/null)

FB_BPB=$("$PYTHON_BIN" -c "
import re
with open('${MASTER_LOG}') as f: c = f.read()
m = re.search(r'RESULT FB.*bpb=([\d.]+)', c)
print(float(m.group(1)) if m else 999)
" 2>/dev/null)

"$PYTHON_BIN" - <<PYEOF
fa = float("${FA_BPB}")
fb = float("${FB_BPB}")
delta = fb - fa
print(f"  FA (current best): {fa:.4f} BPB")
print(f"  FB (H100 candidate): {fb:.4f} BPB")
print(f"  Delta: {delta:+.4f} BPB  ({'FB better' if delta < 0 else 'FA better'})")
print()
if delta < -0.005:
    print("  ✅ VERDICT: Use XSA_START_LAYER=0 + BIGRAM_HASH_BUCKETS=3072 + BIGRAM_HASH_DIM=112")
    print("             Both winner's choices help our SKC model. Lock this in for H100.")
elif delta > 0.005:
    print("  ✅ VERDICT: Keep XSA_START_LAYER=999 + BIGRAM_HASH_BUCKETS=4096 + BIGRAM_HASH_DIM=128")
    print("             Our config is better than the winner's choices on our architecture.")
    print("             Our model is genuinely novel — trust the ablations.")
else:
    print("  ✅ VERDICT: Difference negligible (<0.005 BPB). Use FB config for H100.")
    print("             XSA-all costs nothing and may scale better on H100.")
print()
print("  Next: run orchestrate_h100_experiment.sh with confirmed config.")
PYEOF

log ""
log "Logs:"
log "  FA: recal_FA_current_best_${TS}.log"
log "  FB: recal_FB_h100_candidate_${TS}.log"
log "  Master: recalibration_${TS}.log"
