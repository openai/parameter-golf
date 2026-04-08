#!/bin/bash
# ============================================================================
# SSM BOTTLENECK ABLATION — 5 MIN × 3 CONFIGS
#
# Question: Does the stabilized SSMBottleneck beat CapsuleBank and baseline?
#
# Configs:
#   A — Baseline      : no bottleneck, 8L dim=256
#   B — CapsuleBank   : original softmax-over-time bottleneck (non-causal)
#   C — SSM-v2        : stabilized causal SSM (RMSNorm(h), zero-init C, multi-scale A)
#
# Fixed: 8L dim=256, curriculum 64→256→1024 @5%/20%, XSA=999, seed=42
# Log every 10 steps for fine-grained curve analysis.
# ============================================================================
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR"

TS=$(date +%s)
mkdir -p logs
RESULTS="/tmp/ssm_v2_results_${TS}.txt"

run_config() {
    local NAME=$1
    shift
    local LOG="logs/ssm_v2_${NAME}_${TS}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUNNING CONFIG: $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    RUN_ID="ssm_v2_${NAME}_${TS}" \
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
    FEEDBACK_ENABLED=0  VRL_ENABLED=0 \
    TTT_ENABLED=0  EMA_ENABLED=0  MOE_ENABLED=0 \
    GPTQ_LITE_ENABLED=1  TURBO_QUANT_EXPORT=1  TURBO_QUANT_TRAIN=0  TURBO_QUANT_KV=1 \
    NGRAM_CACHE_ENABLED=1  NGRAM_MAX_ORDER=5 \
    NGRAM_ALPHA_BASE=0.05  NGRAM_ALPHA_SCALE=0.55  NGRAM_ENTROPY_CENTER=4.0 \
    SLIDING_EVAL=1  SLIDING_EVAL_STRIDE=64  TEMP_SCALING=1 \
    TRAIN_LOG_EVERY=10  VAL_BATCH_SIZE=65536  VAL_LOSS_EVERY=0 \
    SEED=42 \
    env "$@" \
    bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

    STEPS=$(grep "^step:" "$LOG" | grep -v "val_loss" | tail -1 | sed 's/step:\([0-9]*\)\/.*/\1/')
    BPB_N=$(grep "ngram_cache" "$LOG" | grep -o 'val_bpb:[0-9.]*' | tail -1 | cut -d: -f2)
    LOSS_F=$(grep "^step:" "$LOG" | grep -v "val_loss" | tail -1 | grep -o 'loss:[0-9.]*' | cut -d: -f2)
    STEPMS=$(grep "^step:" "$LOG" | grep -v "val_loss" | awk -F'step:' '{print $NF}' | sed 's/ms//' | tail -5 | awk '{s+=$1;n++} END{printf "%.0f", s/n}')
    echo "${NAME}|${STEPS:-?}|${LOSS_F:-?}|${BPB_N:-?}|${STEPMS:-?}" >> "$RESULTS"
    echo "  → steps=${STEPS:-?}  loss=${LOSS_F:-?}  bpb_ngram=${BPB_N:-?}  avg_step=${STEPMS:-?}ms"
}

echo "════════════════════════════════════════════════════════"
echo "  SSM v2 ABLATION — 5min × 3 configs"
echo "  8L dim=256 | curriculum 64→256→1024 @5%/20% | seed=42"
echo "════════════════════════════════════════════════════════"

# A: No bottleneck
run_config "A_baseline" \
    CAPSULE_ENABLED=0  SSM_BOTTLENECK=0

# B: CapsuleBank (original)
run_config "B_capsule" \
    CAPSULE_ENABLED=1  SSM_BOTTLENECK=0  KOOPMAN_ENABLED=1

# C: SSM v2 (stabilized: RMSNorm(h) + zero-init C + multi-scale A)
run_config "C_ssm_v2" \
    CAPSULE_ENABLED=1  SSM_BOTTLENECK=1  SSM_BOTTLENECK_STATE_DIM=64

echo ""
echo "════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════════"
python3 - <<'PYEOF'
import os, glob
f = f'/tmp/ssm_v2_results_{os.environ.get("TS","")}.txt'
if not os.path.exists(f):
    files = sorted(glob.glob('/tmp/ssm_v2_results_*.txt'))
    if files: f = files[-1]

labels = {
    'A_baseline': 'A  baseline   no bottleneck              ',
    'B_capsule':  'B  capsule    CapsuleBank + Koopman       ',
    'C_ssm_v2':   'C  SSM-v2     causal, RMSNorm(h), multi-A ',
}

rows = []
for line in open(f):
    parts = line.strip().split('|')
    if len(parts) == 5:
        name, steps, loss, bpb_n, stepms = parts
        try: score = float(bpb_n)
        except: score = 99.0
        rows.append((score, labels.get(name, name), steps, loss, bpb_n, stepms))
rows.sort()

print(f"  {'Config':<44} {'steps':>6}  {'loss':>6}  {'bpb_n':>7}  {'ms/step':>7}")
print(f"  {'─'*44} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
for i, (score, label, steps, loss, bpb_n, stepms) in enumerate(rows):
    star = ' ← BEST' if i == 0 else ''
    print(f"  {label} {steps:>6}  {loss:>6}  {bpb_n:>7}  {stepms:>7}{star}")

if len(rows) >= 2:
    base = next((r[4] for r in rows if 'baseline' in r[1]), None)
    best = rows[0][4]
    if base and best != '?':
        print(f"\n  Delta best vs baseline: {float(best)-float(base):+.4f} BPB")
PYEOF
echo "  Full logs: logs/ssm_v2_*_${TS}.log"
