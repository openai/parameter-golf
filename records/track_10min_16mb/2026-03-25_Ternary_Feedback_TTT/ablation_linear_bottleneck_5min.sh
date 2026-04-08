#!/bin/bash
set -euo pipefail
EXPDIR="$(cd "$(dirname "$0")" && pwd)"; cd "$EXPDIR"
TS=$(date +%s); mkdir -p logs
RESULTS="/tmp/linbottle_results_${TS}.txt"

run_config() {
    local NAME=$1; shift
    local LOG="logs/lb_${NAME}_${TS}.log"
    echo ""; echo "━━━ CONFIG: $NAME ━━━"
    env ARCHITECTURE=skc NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=4 \
    VOCAB_SIZE=1024 SKC_BLOCK_SIZE=16 SKC_NUM_CAPSULES=16 SKC_CAPSULE_DIM=64 SKC_CONV_KERNEL=4 \
    XSA_START_LAYER=999 BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=3072 BIGRAM_HASH_DIM=112 \
    ENGRAM_NUM_HEADS=4 ENGRAM_NUM_ORDERS=3 ENGRAM_INJECT_LAYER=1 PARTIAL_ROPE_DIMS=16 LN_SCALE_DAMPING=1 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=4 MLX_MAX_MICROBATCH_TOKENS=8192 MLX_EAGER_EVAL=1 \
    MAX_WALLCLOCK_SECONDS=300 ITERATIONS=1000000 WARMUP_STEPS=5 WARMDOWN_FRACTION=0.3 \
    CURRICULUM_ENABLED=0 MATRIX_LR=0.02 SCALAR_LR=0.015 TIED_EMBED_LR=0.035 \
    MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_STEPS=0 MUON_BACKEND_STEPS=5 \
    MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
    LAWA_ENABLED=1 LAWA_K=5 LAWA_FREQ=100 SWA_ENABLED=1 SWA_EVERY=50 SMEARGATE_ENABLED=1 TKO_ENABLED=0 \
    FEEDBACK_ENABLED=0 VRL_ENABLED=0 TTT_ENABLED=0 EMA_ENABLED=0 MOE_ENABLED=0 \
    GPTQ_LITE_ENABLED=1 TURBO_QUANT_EXPORT=0 TURBO_QUANT_TRAIN=0 TURBO_QUANT_KV=1 \
    NGRAM_CACHE_ENABLED=1 NGRAM_MAX_ORDER=5 NGRAM_ALPHA_BASE=0.05 NGRAM_ALPHA_SCALE=0.55 NGRAM_ENTROPY_CENTER=4.0 \
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 TEMP_SCALING=1 \
    TRAIN_LOG_EVERY=20 VAL_BATCH_SIZE=65536 VAL_LOSS_EVERY=0 SEED=42 \
    RUN_ID="lb_${NAME}_${TS}" "$@" bash run_mlx_reasoner.sh 2>&1 | tee "$LOG"

    STEPS=$(grep "^step:" "$LOG" | grep -v val_loss | tail -1 | sed 's/step:\([0-9]*\)\/.*/\1/')
    BPB=$(grep "ngram_cache" "$LOG" | grep -o 'val_bpb:[0-9.]*' | tail -1 | cut -d: -f2)
    LOSS=$(grep "^step:" "$LOG" | grep -v val_loss | tail -1 | grep -o 'loss:[0-9.]*' | cut -d: -f2)
    MS=$(grep "^step:" "$LOG" | grep -v val_loss | awk -F'step:' '{print $NF}' | sed 's/ms//' | tail -10 | awk '{s+=$1;n++} END{printf "%.0f",s/n}')
    echo "${NAME}|${STEPS:-?}|${LOSS:-?}|${BPB:-?}|${MS:-?}" >> "$RESULTS"
    echo "  → steps=${STEPS:-?}  loss=${LOSS:-?}  bpb=${BPB:-?}  ms/step=${MS:-?}"
}

echo "════════════════════════════════════════════"
echo "  Linear Bottleneck Ablation | seq=256 | 5min"
echo "════════════════════════════════════════════"

run_config "A_baseline"    CAPSULE_ENABLED=0
run_config "B_linear_r32"  CAPSULE_ENABLED=1 LINEAR_BOTTLENECK=1 LINEAR_BOTTLENECK_RANK=32
run_config "C_linear_r64"  CAPSULE_ENABLED=1 LINEAR_BOTTLENECK=1 LINEAR_BOTTLENECK_RANK=64
run_config "D_linear_r128" CAPSULE_ENABLED=1 LINEAR_BOTTLENECK=1 LINEAR_BOTTLENECK_RANK=128

echo ""
echo "════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════"
python3 - <<'PYEOF'
import os, glob
f = f'/tmp/linbottle_results_{os.environ.get("TS","")}.txt'
if not os.path.exists(f):
    files = sorted(glob.glob('/tmp/linbottle_results_*.txt'))
    if files: f = files[-1]
labels = {
    'A_baseline':    'A  baseline       no bottleneck     ',
    'B_linear_r32':  'B  linear rank=32                   ',
    'C_linear_r64':  'C  linear rank=64                   ',
    'D_linear_r128': 'D  linear rank=128                  ',
}
rows = []
for line in open(f):
    p = line.strip().split('|')
    if len(p)==5:
        name,steps,loss,bpb,ms = p
        try: score=float(bpb)
        except: score=99.0
        rows.append((score,labels.get(name,name),steps,loss,bpb,ms))
rows.sort()
print(f"  {'Config':<38} {'steps':>6}  {'loss':>6}  {'bpb':>7}  {'ms/stp':>6}")
print(f"  {'─'*38} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")
for i,(score,label,steps,loss,bpb,ms) in enumerate(rows):
    star=' ← BEST' if i==0 else ''
    print(f"  {label} {steps:>6}  {loss:>6}  {bpb:>7}  {ms:>6}{star}")
base=next((r[4] for r in rows if 'baseline' in r[1]),None)
if base and rows[0][4]!='?':
    print(f"\n  Delta best vs baseline: {float(rows[0][4])-float(base):+.4f} BPB")
PYEOF
