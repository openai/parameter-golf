#!/bin/bash
# Validate UNet capsule skip + Koopman speculation improvements
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR" || exit 1
PY=/opt/homebrew/Cellar/python@3.12/3.12.3/Frameworks/Python.framework/Versions/3.12/bin/python3.12
TS=$(date +%s)

run_exp() {
    local name=$1; shift
    local logfile="${EXPDIR}/exp_unet_${name}_${TS}.log"
    echo "====== START: ${name}  $(date) ======"
    echo "START: ${name}  $(date)" > "$logfile"
    env \
    ARCHITECTURE=skc NUM_LAYERS=6 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 SEED=42 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=8192 MAX_WALLCLOCK_SECONDS=300 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=128 \
    LAWA_ENABLED=1 LAWA_K=5 SWA_ENABLED=1 SMEARGATE_ENABLED=1 TKO_ENABLED=0 \
    SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=8 SKC_BLOCK_SIZE=16 \
    FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 \
    ITERATIONS=100000 MATRIX_LR=0.02 SCALAR_LR=0.015 \
    VRL_ENABLED=0 BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3 TTT_ENABLED=0 \
    NGRAM_CACHE_ENABLED=1 XSA_START_LAYER=999 TEMP_SCALING=0 EMA_ENABLED=0 \
    MUON_MOMENTUM_WARMUP_STEPS=0 \
    "$@" bash run_mlx_reasoner.sh >> "$logfile" 2>&1
    $PY -c "
import re
with open('${logfile}') as f: c=f.read()
bpbs=re.findall(r'val_bpb:([\d.]+)',c)
steps=re.findall(r'step:(\d+)/',c)
params=re.findall(r'model_params:(\d+)',c)
print(f'  -> RESULT ${name}: BPB={float(bpbs[-1]):.4f}  steps={steps[-1]}  params={params[0] if params else \"?\"}' if bpbs else f'  -> ${name}: NO BPB')
" 2>&1
}

echo "=== UNet Validation START: $(date) ==="

# 1. Best config baseline (no UNet caps, no speculation) — should match CC_best_v2=2.064
run_exp "DA_best_baseline"

# 2. Best config + UNet capsule skips only (CAPSULE_ENABLED=0, skips still work via enc_caps)
#    enc_caps is collected in encoder, symmetric skip blends in decoder
#    No CapsuleBank (CAPSULE_ENABLED=0) so bottleneck_caps=None, speculation can't fire
run_exp "DB_unet_caps_skip"  CAPSULE_ENABLED=0

# 3. Best config + CapsuleBank bottleneck only (no speculation)
run_exp "DC_capsule_bank"  CAPSULE_ENABLED=1 KOOPMAN_SPECULATOR_ENABLED=0

# 4. Best config + CapsuleBank + Koopman speculation (full UNet Koopman)
run_exp "DD_full_unet_koopman"  \
    CAPSULE_ENABLED=1 \
    KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=3 \
    KOOPMAN_SPECULATOR_WEIGHT=0.01

# 5. Same as DD but higher speculation weight
run_exp "DE_koopman_w05"  \
    CAPSULE_ENABLED=1 \
    KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=3 \
    KOOPMAN_SPECULATOR_WEIGHT=0.05

# 6. Same as DD but lighter speculation (1 step instead of 3)
run_exp "DF_koopman_1step"  \
    CAPSULE_ENABLED=1 \
    KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=1 \
    KOOPMAN_SPECULATOR_WEIGHT=0.01

# 7. Second seed for best config to establish variance baseline
run_exp "DG_best_seed1337"  SEED=1337

# 8. Second seed for full UNet Koopman  
run_exp "DH_unet_koopman_s1337"  \
    SEED=1337 CAPSULE_ENABLED=1 \
    KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=3 \
    KOOPMAN_SPECULATOR_WEIGHT=0.01

echo "=== UNet Validation COMPLETE: $(date) ==="
