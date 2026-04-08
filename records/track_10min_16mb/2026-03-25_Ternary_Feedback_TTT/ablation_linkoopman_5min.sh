#!/bin/bash
# ============================================================================
# LINEAR-ATTENTION KOOPMAN ABLATION — 5 MIN × 3 CONFIGS
#
# Question: Does LinearAttnKoopman (full-matrix Koopman state) beat
#           diagonal SSM and baseline at the bottleneck position?
#
# Configs:
#   A — baseline       : no bottleneck
#   B — SSM            : diagonal Koopman (state=64)
#   C — LinKoopman     : full-matrix Koopman (H=4, d=16, state=H×d×d=1024 floats)
#
# Fixed: seq=256 (fast ablations), 8L dim=256, no curriculum, seed=42
# seq=256 keeps step times short; all configs see same tokens for fair comparison.
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
mkdir -p logs
RESULTS="/tmp/linkoopman_results_${TS}.txt"

BASE_ENV=(
    ARCHITECTURE=skc
    NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4
    VOCAB_SIZE=1024
    SKC_BLOCK_SIZE=16 SKC_NUM_CAPSULES=16 SKC_CAPSULE_DIM=64 SKC_CONV_KERNEL=4
    XSA_START_LAYER=999
    BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=3072 BIGRAM_HASH_DIM=112
    ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=3 ENGRAM_INJECT_LAYER=1
    PARTIAL_ROPE_DIMS=16 LN_SCALE_DAMPING=1
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=4
    MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1
    MAX_WALLCLOCK_SECONDS=300 ITERATIONS=1000000
    WARMUP_STEPS=5 WARMDOWN_FRACTION=0.3
    CURRICULUM_ENABLED=0
    MATRIX_LR=0.02 SCALAR_LR=0.015 TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_STEPS=0 MUON_BACKEND_STEPS=5
    MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
    LAWA_ENABLED=1 LAWA_K=5 LAWA_FREQ=100
    SWA_ENABLED=1 SWA_EVERY=50 SMEARGATE_ENABLED=1 TKO_ENABLED=0
    FEEDBACK_ENABLED=0 VRL_ENABLED=0 TTT_ENABLED=0 EMA_ENABLED=0 MOE_ENABLED=0
    GPTQ_LITE_ENABLED=1 TURBO_QUANT_EXPORT=0 TURBO_QUANT_TRAIN=0 TURBO_QUANT_KV=1
    NGRAM_CACHE_ENABLED=1 NGRAM_MAX_ORDER=5
    NGRAM_ALPHA_BASE=0.05 NGRAM_ALPHA_SCALE=0.55 NGRAM_ENTROPY_CENTER=4.0
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 TEMP_SCALING=1
    TRAIN_LOG_EVERY=20 VAL_BATCH_SIZE=65536 VAL_LOSS_EVERY=0
    SEED=42
)

run_config() {
    local NAME=$1; shift
    local LOG="logs/lko_${NAME}_${TS}.log"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CONFIG: $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    env "${BASE_ENV[@]}" RUN_ID="lko_${NAME}_${TS}" "$@" \
        bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

    STEPS=$(grep "^step:" "$LOG" | grep -v val_loss | tail -1 | sed 's/step:\([0-9]*\)\/.*/\1/')
    BPB=$(grep "ngram_cache" "$LOG" | grep -o 'val_bpb:[0-9.]*' | tail -1 | cut -d: -f2)
    LOSS=$(grep "^step:" "$LOG" | grep -v val_loss | tail -1 | grep -o 'loss:[0-9.]*' | cut -d: -f2)
    MS=$(grep "^step:" "$LOG" | grep -v val_loss | awk -F'step:' '{print $NF}' | sed 's/ms//' | \
         tail -10 | awk '{s+=$1;n++} END{printf "%.0f",s/n}')
    echo "${NAME}|${STEPS:-?}|${LOSS:-?}|${BPB:-?}|${MS:-?}" >> "$RESULTS"
    echo "  → steps=${STEPS:-?}  loss=${LOSS:-?}  bpb=${BPB:-?}  ms/step=${MS:-?}"
}

echo "════════════════════════════════════════════════════"
echo "  LinKoopman Ablation | seq=256 | 5min × 3 configs"
echo "  8L dim=256 | no curriculum | seed=42"
echo "════════════════════════════════════════════════════"

run_config "A_baseline" \
    CAPSULE_ENABLED=0

run_config "B_ssm64" \
    CAPSULE_ENABLED=1 SSM_BOTTLENECK=1 SSM_BOTTLENECK_STATE_DIM=64 LINKOOPMAN_ENABLED=0

run_config "C_linkoopman" \
    CAPSULE_ENABLED=1 LINKOOPMAN_ENABLED=1 LINKOOPMAN_HEADS=4 LINKOOPMAN_HEAD_DIM=16

echo ""
echo "════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════"
python3 - <<'PYEOF'
import os, glob
f = f'/tmp/linkoopman_results_{os.environ.get("TS","")}.txt'
if not os.path.exists(f):
    files = sorted(glob.glob('/tmp/linkoopman_results_*.txt'))
    if files: f = files[-1]

labels = {
    'A_baseline':   'A  baseline     no bottleneck                 ',
    'B_ssm64':      'B  SSM-diag     diagonal Koopman (state=64)   ',
    'C_linkoopman': 'C  LinKoopman   full-matrix Koopman (H=4×d=16)',
}
rows = []
for line in open(f):
    p = line.strip().split('|')
    if len(p) == 5:
        name, steps, loss, bpb, ms = p
        try: score = float(bpb)
        except: score = 99.0
        rows.append((score, labels.get(name, name), steps, loss, bpb, ms))
rows.sort()

print(f"  {'Config':<48} {'steps':>6}  {'loss':>6}  {'bpb':>7}  {'ms/stp':>6}")
print(f"  {'─'*48} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")
for i, (score, label, steps, loss, bpb, ms) in enumerate(rows):
    star = ' ← BEST' if i == 0 else ''
    print(f"  {label} {steps:>6}  {loss:>6}  {bpb:>7}  {ms:>6}{star}")
if len(rows) >= 2:
    base = next((r[4] for r in rows if 'baseline' in r[1]), None)
    if base and rows[0][4] != '?':
        print(f"\n  Delta best vs baseline: {float(rows[0][4])-float(base):+.4f} BPB")
PYEOF
echo "  Logs: logs/lko_*_${TS}.log"
