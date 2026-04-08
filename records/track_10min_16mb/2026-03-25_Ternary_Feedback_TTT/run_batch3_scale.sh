#!/bin/bash
# Batch 3: block_size sweep + scaling experiments
# ONE at a time, strictly sequential. NEVER parallel.
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR" || exit 1
PY=/opt/homebrew/Cellar/python@3.12/3.12.3/Frameworks/Python.framework/Versions/3.12/bin/python3.12
TS=$(date +%s)

run_exp() {
    local name=$1; shift
    local logfile="${EXPDIR}/exp_b3_${name}_${TS}.log"
    echo "====== START: ${name}  $(date) ======"
    echo "START: ${name}  $(date)" >> batch3_master.log
    echo "START: ${name}  $(date)" > "$logfile"
    env \
    ARCHITECTURE=skc NUM_LAYERS=6 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 SEED=42 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=8192 MAX_WALLCLOCK_SECONDS=300 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=128 \
    LAWA_ENABLED=1 LAWA_K=5 SWA_ENABLED=1 SMEARGATE_ENABLED=1 TKO_ENABLED=0 \
    SKC_CAPSULE_DIM=32 SKC_NUM_CAPSULES=8 SKC_BLOCK_SIZE=16 \
    FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 \
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 ITERATIONS=100000 \
    MATRIX_LR=0.035 SCALAR_LR=0.025 \
    VRL_ENABLED=0 BIGRAM_HASH_ENABLED=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
    XSA_START_LAYER=999 TEMP_SCALING=0 EMA_ENABLED=0 \
    KOOPMAN_SPECULATOR_ENABLED=0 \
    "$@" bash run_mlx_reasoner.sh >> "$logfile" 2>&1
    local bpb
    bpb=$($PY -c "
import re
with open('${logfile}') as f: c=f.read()
bpbs=re.findall(r'val_bpb:([\d.]+)',c)
steps=re.findall(r'step:(\d+)/',c)
print(f'BPB={float(bpbs[-1]):.4f}  steps={steps[-1]}' if bpbs else 'NO_BPB')
" 2>&1)
    echo "  -> RESULT ${name}: ${bpb}"
    echo "final_result ${name}: ${bpb}" >> batch3_master.log
}

echo "=== BATCH-3 START: $(date) ==="
echo "BATCH3_START: $(date)" >> batch3_master.log

# Base for this batch: 6L dim=128, TKO=0, block_size=16 (best known so far)
# All experiments vary one thing from this base.

# ── WHT block_size=8 (smaller than our best block_size=16) ───────────────────
run_exp "EA_wht8"    SKC_BLOCK_SIZE=8

# ── Capsule routing: more capsules with block_size=16 (1:1 ratio) ────────────
run_exp "EB_caps16"  SKC_NUM_CAPSULES=16 SKC_CAPSULE_DIM=32

# ── block_size=8 + 16 capsules (1:1 sequency:capsule ratio at block=8) ───────
run_exp "EC_wht8_caps16"  SKC_BLOCK_SIZE=8 SKC_NUM_CAPSULES=16

# ── Engram + block_size=16 (stack our two best findings) ─────────────────────
run_exp "ED_engram16"   BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3

# ── Low LR + block_size=16 ───────────────────────────────────────────────────
run_exp "EE_lr_low16"   MATRIX_LR=0.02 SCALAR_LR=0.015

# ── SCALE: 6L dim=256 (~0.7MB) ───────────────────────────────────────────────
run_exp "EF_dim256"  \
    MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 \
    SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=8 \
    TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256

# ── SCALE: 8L dim=256 (~1.0MB) ───────────────────────────────────────────────
run_exp "EG_8L_dim256"  \
    NUM_LAYERS=8 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 \
    SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=8 \
    TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256

# ── SCALE: 8L dim=384 (safe size — avoids OOM on Mac) ────────────────────────
# dim=512 with large batches risks OOM; dim=384 is the safe big-model test
run_exp "EH_8L_dim384"  \
    NUM_LAYERS=8 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 \
    SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=16 \
    TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256

# ── Koopman multistep speculation (user request: DF was 1-step; try 3-step) ──
# DF_koopman_1step=1.951 was −0.007 vs baseline; try more steps to see if
# multistep pressure actually helps the capsule bank dynamics
run_exp "EJ_koop_3step"  \
    KOOPMAN_SPECULATOR_ENABLED=1 \
    KOOPMAN_SPECULATOR_STEPS=3 \
    KOOPMAN_SPECULATOR_WEIGHT=0.01

# ── Koopman multistep with lower weight (less pressure, more signal) ─────────
run_exp "EK_koop_3step_w001"  \
    KOOPMAN_SPECULATOR_ENABLED=1 \
    KOOPMAN_SPECULATOR_STEPS=3 \
    KOOPMAN_SPECULATOR_WEIGHT=0.001

# ── BEST SMALL COMBO: block16 + engram + low_lr + caps16 + ngram ─────────────
run_exp "EL_best_small"  \
    BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3 \
    MATRIX_LR=0.02 SCALAR_LR=0.015 \
    SKC_NUM_CAPSULES=16 SKC_CAPSULE_DIM=32 \
    NGRAM_CACHE_ENABLED=1

echo "=== BATCH-3 COMPLETE: $(date) ==="
echo "BATCH3_COMPLETE: $(date)" >> batch3_master.log
