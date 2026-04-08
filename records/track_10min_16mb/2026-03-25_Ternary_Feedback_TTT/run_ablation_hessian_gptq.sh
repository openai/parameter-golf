#!/bin/bash
# ============================================================================
# ABLATION: HESSIAN_TERNARY_GPTQ on vs off
#
# Two 5-min runs, identical training, only quantization method differs at export.
#
#  GA — HESSIAN_TERNARY_GPTQ=1  (Full Hessian GPTQ, AR self-calibration)
#  GB — HESSIAN_TERNARY_GPTQ=0  (Simple percentile search via gptq_lite)
#
# Both use identical training state — only the final quantization pass differs.
# This isolates the exact question: does Hessian-guided ternary quantization
# produce a better-scoring artifact than simple percentile clipping?
#
# Usage: bash run_ablation_hessian_gptq.sh
# ============================================================================
set -uo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/bin/python3}
[[ ! -f "$PYTHON_BIN" ]] && PYTHON_BIN=python3

TS=$(date +%s)
MASTER_LOG="${EXPDIR}/ablation_hessian_gptq_${TS}.log"

log() { echo "$*" | tee -a "$MASTER_LOG"; }

# ── Shared base config (best proven settings, 5-min budget) ──────────────────
BASE_ARGS=(
    ARCHITECTURE=skc
    NUM_LAYERS=8  MODEL_DIM=256  NUM_HEADS=4  NUM_KV_HEADS=2  MLP_MULT=4
    VOCAB_SIZE=1024
    SKC_BLOCK_SIZE=16  SKC_NUM_CAPSULES=16  SKC_CAPSULE_DIM=64  SKC_CONV_KERNEL=4
    XSA_START_LAYER=0
    BIGRAM_HASH_ENABLED=1  BIGRAM_HASH_BUCKETS=3072  BIGRAM_HASH_DIM=112
    ENGRAM_NUM_HEADS=4  ENGRAM_NUM_ORDERS=3  ENGRAM_INJECT_LAYER=1
    PARTIAL_ROPE_DIMS=16  LN_SCALE_DAMPING=1
    TRAIN_SEQ_LEN=512  TRAIN_BATCH_TOKENS=8192  GRAD_ACCUM_STEPS=4
    MLX_MAX_MICROBATCH_TOKENS=8192  MLX_EAGER_EVAL=1
    MAX_WALLCLOCK_SECONDS=300  ITERATIONS=100000
    WARMUP_STEPS=5  WARMDOWN_FRACTION=0.5
    CURRICULUM_ENABLED=1
    CURRICULUM_PHASE1_SEQ=64  CURRICULUM_PHASE2_SEQ=256
    CURRICULUM_PHASE1_FRAC=0.05  CURRICULUM_PHASE2_FRAC=0.20
    STOCHASTIC_DEPTH_PROB=0
    MATRIX_LR=0.02  SCALAR_LR=0.015  TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.95  MUON_MOMENTUM_WARMUP_STEPS=0  MUON_BACKEND_STEPS=5
    MUON_WD=0.04  ADAM_WD=0.04  GRAD_CLIP_NORM=0.3
    LAWA_ENABLED=1  LAWA_K=10  LAWA_FREQ=100
    SWA_ENABLED=1  SWA_EVERY=50  SMEARGATE_ENABLED=1  TKO_ENABLED=0
    FEEDBACK_ENABLED=0  CAPSULE_ENABLED=0  VRL_ENABLED=0
    TTT_ENABLED=0  EMA_ENABLED=0  MOE_ENABLED=0
    TURBO_QUANT_EXPORT=1  TURBO_QUANT_TRAIN=0  TURBO_QUANT_KV=1
    BITNET_GROUP_SIZE=128
    SELECTIVE_PRUNING=1  SELECTIVE_PRUNING_TARGET_MB=15.5
    NGRAM_CACHE_ENABLED=1  NGRAM_MAX_ORDER=5
    NGRAM_ALPHA_BASE=0.05  NGRAM_ALPHA_SCALE=0.55  NGRAM_ENTROPY_CENTER=4.0
    SLIDING_EVAL=1  SLIDING_EVAL_STRIDE=64  TEMP_SCALING=1
    TRAIN_LOG_EVERY=50  VAL_BATCH_SIZE=65536
    SEED=42
)

parse_result() {
    local logfile=$1
    local batch_tokens=$2
    "$PYTHON_BIN" -c "
import re
with open('${logfile}') as f: c = f.read()
steps = re.findall(r'step:(\d+)/', c)
# Best BPB: prefer ngram, then sliding, then val
for pat in [r'ngram_cache.*?val_bpb:([\d.]+)', r'final_sliding.*?val_bpb:([\d.]+)', r'val_bpb:([\d.]+)']:
    m = re.findall(pat, c)
    if m:
        bpb = float(m[-1]); break
else: bpb = None
sizes = re.findall(r'artifact:([\d.]+)MB', c)
gptq_time = re.findall(r'hessian_ternary_gptq:done in ([\d.]+)s', c)
toks = int(steps[-1]) * ${batch_tokens} / 1e6 if steps else 0
print(f'  steps   : {steps[-1] if steps else \"?\"}')
print(f'  tokens  : {toks:.1f}M')
print(f'  bpb     : {bpb if bpb else \"?\"}')
print(f'  artifact: {sizes[-1] if sizes else \"?\"}')
print(f'  gptq_t  : {gptq_time[0]+\"s\" if gptq_time else \"n/a (off)\"}')
" 2>/dev/null
}

log "========================================================"
log "  ABLATION: HESSIAN_TERNARY_GPTQ on vs off"
log "  5-min × 2 runs  |  $(date)"
log "========================================================"

# ── GA: HESSIAN_TERNARY_GPTQ=1 (Hessian GPTQ, our planned H100 config) ──────
LOG_GA="${EXPDIR}/ablation_GA_hessian_on_${TS}.log"
log ""
log ">>> GA: HESSIAN_TERNARY_GPTQ=1  (Full Hessian + AR self-calibration)"
env "${BASE_ARGS[@]}" \
    HESSIAN_TERNARY_GPTQ=1  GPTQ_LITE_ENABLED=0 \
    RUN_ID="ablation_GA_hessian_on_${TS}" \
    bash run_mlx_reasoner.sh 2>&1 | tee "$LOG_GA"
log "GA complete"
log "$(parse_result "$LOG_GA" 8192)"
echo "RESULT_GA: $(parse_result "$LOG_GA" 8192)" >> "$MASTER_LOG"

# ── GB: HESSIAN_TERNARY_GPTQ=0 (simple percentile clipping fallback) ─────────
LOG_GB="${EXPDIR}/ablation_GB_hessian_off_${TS}.log"
log ""
log ">>> GB: HESSIAN_TERNARY_GPTQ=0  (percentile clipping only)"
env "${BASE_ARGS[@]}" \
    HESSIAN_TERNARY_GPTQ=0  GPTQ_LITE_ENABLED=1 \
    RUN_ID="ablation_GB_hessian_off_${TS}" \
    bash run_mlx_reasoner.sh 2>&1 | tee "$LOG_GB"
log "GB complete"
log "$(parse_result "$LOG_GB" 8192)"
echo "RESULT_GB: $(parse_result "$LOG_GB" 8192)" >> "$MASTER_LOG"

# ── Verdict ───────────────────────────────────────────────────────────────────
log ""
log "========================================================"
log "  VERDICT"
log "========================================================"
"$PYTHON_BIN" - <<PYEOF
import re
def get_bpb(f):
    with open(f) as fp: c = fp.read()
    for pat in [r'ngram_cache.*?val_bpb:([\d.]+)', r'final_sliding.*?val_bpb:([\d.]+)', r'val_bpb:([\d.]+)']:
        m = re.findall(pat, c)
        if m: return float(m[-1])
    return None

ga = get_bpb("${LOG_GA}")
gb = get_bpb("${LOG_GB}")

if ga and gb:
    delta = ga - gb
    print(f"  GA (Hessian ON) : {ga:.4f} BPB")
    print(f"  GB (Hessian OFF): {gb:.4f} BPB")
    print(f"  Delta           : {delta:+.4f} BPB  ({'Hessian helps' if delta < 0 else 'Hessian hurts' if delta > 0 else 'no difference'})")
    print()
    if abs(delta) < 0.002:
        print("  → Negligible difference. Either config fine for H100.")
    elif delta < 0:
        print(f"  → Hessian GPTQ is better by {abs(delta):.4f} BPB. Keep HESSIAN_TERNARY_GPTQ=1.")
    else:
        print(f"  → Percentile clipping is better by {abs(delta):.4f} BPB.")
        print(f"  → Set HESSIAN_TERNARY_GPTQ=0, GPTQ_LITE_ENABLED=1 in H100 run script.")
else:
    print("  Could not parse BPB from one or both logs.")
PYEOF

log ""
log "Logs: $(basename $LOG_GA)  |  $(basename $LOG_GB)"
