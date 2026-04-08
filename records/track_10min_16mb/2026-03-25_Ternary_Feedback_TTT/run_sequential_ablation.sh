#!/bin/bash
# Sequential ablation — ONE experiment at a time, never parallel.
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR" || exit 1
PY=/opt/homebrew/Cellar/python@3.12/3.12.3/Frameworks/Python.framework/Versions/3.12/bin/python3.12
TS=$(date +%s)

run_exp() {
    local name=$1; shift
    local logfile="${EXPDIR}/exp_seq_${name}_${TS}.log"
    echo "====== START $name @ $(date) ======"
    echo "====== START $name @ $(date) ======" > "$logfile"
    env \
    ARCHITECTURE=skc NUM_LAYERS=4 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 SEED=42 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=8192 MAX_WALLCLOCK_SECONDS=300 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=128 \
    LAWA_ENABLED=1 LAWA_K=5 SWA_ENABLED=1 SMEARGATE_ENABLED=1 TKO_ENABLED=1 \
    SKC_CAPSULE_DIM=32 SKC_NUM_CAPSULES=8 FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 \
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 \
    ITERATIONS=100000 \
    MATRIX_LR=0.035 SCALAR_LR=0.025 \
    VRL_ENABLED=0 BIGRAM_HASH_ENABLED=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
    XSA_START_LAYER=999 TEMP_SCALING=0 \
    "$@" bash run_mlx_reasoner.sh >> "$logfile" 2>&1
    echo "====== END $name @ $(date) ======" | tee -a "$logfile"
    $PY -c "
import re
with open('${logfile}') as f: c = f.read()
bpbs = re.findall(r'val_bpb:([\d.]+)', c)
steps = re.findall(r'step:(\d+)/', c)
if bpbs:
    print(f'  -> RESULT ${name}: BPB={float(bpbs[-1]):.4f}  steps={steps[-1] if steps else \"?\"}')
else:
    lines = c.strip().split(chr(10))
    print(f'  -> ${name}: no BPB. Last: {lines[-1][:120]}')
" 2>&1
}

echo "=== SEQUENTIAL ABLATION START: $(date) ==="

# B: 6 layers (matched-param SKC)
run_exp "B_6layers"        NUM_LAYERS=6

# F: Curriculum OFF
run_exp "F_curriculum_off" CURRICULUM_ENABLED=0

# G: LAWA+SWA OFF
run_exp "G_lawa_off"       LAWA_ENABLED=0 SWA_ENABLED=0

# H: Smeargate OFF
run_exp "H_smeargate_off"  SMEARGATE_ENABLED=0

# I: LR high
run_exp "I_lr_high"        MATRIX_LR=0.05 SCALAR_LR=0.04

# J: LR low
run_exp "J_lr_low"         MATRIX_LR=0.02 SCALAR_LR=0.015

# K: 6 layers seed=1337
run_exp "K_6layers_s1337"  NUM_LAYERS=6 SEED=1337

# L: Weight sharing
run_exp "L_weight_share"   NUM_LAYERS=6 WEIGHT_SHARING=1

# M: Feedback ON
run_exp "M_feedback"       FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 CAPSULE_ENABLED=1

# N: Engram ON
run_exp "N_engram"         BIGRAM_HASH_ENABLED=1

# O: Inside-out training
run_exp "O_inside_out"     INSIDE_OUT_TRAINING=1 CAPSULE_ENABLED=1

# P: DEQ feedback
run_exp "P_deq"            DEQ_FEEDBACK=1 CAPSULE_ENABLED=1

# Q: 6L + feedback
run_exp "Q_6L_feedback"    NUM_LAYERS=6 FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 CAPSULE_ENABLED=1

echo "=== SEQUENTIAL ABLATION COMPLETE: $(date) ==="
