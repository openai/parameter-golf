#!/bin/bash
# Batch 2 ablations — every untested feature. ONE at a time, strictly sequential.
EXPDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXPDIR" || exit 1
PY=/opt/homebrew/Cellar/python@3.12/3.12.3/Frameworks/Python.framework/Versions/3.12/bin/python3.12
TS=$(date +%s)

run_exp() {
    local name=$1; shift
    local logfile="${EXPDIR}/exp_b2_${name}_${TS}.log"
    echo "====== START: ${name}  $(date) ======"
    echo "START: ${name}  $(date)" >> missing_ablations_master.log
    echo "START: ${name}  $(date)" > "$logfile"
    env \
    ARCHITECTURE=skc NUM_LAYERS=6 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 SEED=42 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=8192 MAX_WALLCLOCK_SECONDS=300 \
    CURRICULUM_ENABLED=1 CURRICULUM_PHASE1_SEQ=64 CURRICULUM_PHASE2_SEQ=128 \
    LAWA_ENABLED=1 LAWA_K=5 SWA_ENABLED=1 SMEARGATE_ENABLED=1 TKO_ENABLED=1 \
    SKC_CAPSULE_DIM=32 SKC_NUM_CAPSULES=8 SKC_BLOCK_SIZE=64 \
    FEEDBACK_ENABLED=0 CAPSULE_ENABLED=0 \
    SLIDING_EVAL=1 SLIDING_EVAL_STRIDE=32 ITERATIONS=100000 \
    MATRIX_LR=0.035 SCALAR_LR=0.025 \
    VRL_ENABLED=0 BIGRAM_HASH_ENABLED=0 TTT_ENABLED=0 NGRAM_CACHE_ENABLED=0 \
    XSA_START_LAYER=999 TEMP_SCALING=0 EMA_ENABLED=0 \
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
    echo "final_result ${name}: ${bpb}" >> missing_ablations_master.log
}

echo "=== BATCH-2 ABLATION START: $(date) ==="
echo "BATCH2_START: $(date)" >> missing_ablations_master.log

# NOTE: Base is 6L, TKO=1, no engram/VRL. Vary one thing at a time.

# ── WHT block size ────────────────────────────────────────────────────────────
run_exp "BA_wht16"     SKC_BLOCK_SIZE=16
run_exp "BB_wht32"     SKC_BLOCK_SIZE=32
run_exp "BC_wht128"    SKC_BLOCK_SIZE=128

# ── Capsule carry decay (Koopman eigenvalue gate) ─────────────────────────────
run_exp "BD_carry_low"  CAPSULE_ENABLED=1 CAPSULE_CARRY_DECAY=0.5
run_exp "BE_carry_high" CAPSULE_ENABLED=1 CAPSULE_CARRY_DECAY=0.95
run_exp "BF_carry_off"  CAPSULE_ENABLED=1 CAPSULE_CARRY_ENABLED=0

# ── Muon momentum warmup (currently 1500 steps; at 5min=~1500 steps that's 100% warmup) ──
run_exp "BG_muon_warm300"  NUM_LAYERS=6 MUON_MOMENTUM_WARMUP_STEPS=300
run_exp "BH_muon_warm0"    NUM_LAYERS=6 MUON_MOMENTUM_WARMUP_STEPS=0

# ── XSA (cross-sequence attention) ────────────────────────────────────────────
run_exp "BI_xsa4"   XSA_START_LAYER=4
run_exp "BJ_xsa2"   XSA_START_LAYER=2

# ── VRL (vector recurrent learning) only ─────────────────────────────────────
run_exp "BK_vrl_only"  VRL_ENABLED=1

# ── LN scale damping only ────────────────────────────────────────────────────
run_exp "BL_ln_damp"   LN_SCALE_DAMPING=1

# ── Engram orders ─────────────────────────────────────────────────────────────
run_exp "BM_engram_orders3"  BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3
run_exp "BN_engram_orders1"  BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=1

# ── TTT at eval (test-time training, scope=all means applies to SKC) ──────────
run_exp "BO_ttt"  TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=1 TTT_SCOPE=all

# ── Ngram cache (free eval-time BPB boost) ───────────────────────────────────
run_exp "BP_ngram"  NGRAM_CACHE_ENABLED=1

# ── EMA eval (apply EMA shadow weights at val time) ──────────────────────────
run_exp "BQ_ema_eval"  EMA_ENABLED=1 EMA_EVAL_APPLY=1

# ── Koopman speculation (needs CAPSULE+FEEDBACK fixed) ────────────────────────
run_exp "BR_koop_spec"  \
    CAPSULE_ENABLED=1 FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 \
    KOOPMAN_SPECULATOR_ENABLED=1 KOOPMAN_SPECULATOR_STEPS=3 \
    KOOPMAN_SPECULATOR_WEIGHT=0.01

# ── Self-distill KL (consistency between pass-0 and final hidden) ─────────────
run_exp "BS_self_distill"  \
    FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 \
    SELF_DISTILL_KL_WEIGHT=0.1

# ── Stochastic depth (currently 0.0 in base; re-enable original 0.2) ─────────
run_exp "BT_stoch_depth"   STOCHASTIC_DEPTH_PROB=0.2

# ── Ternary noise scale (currently 0.0 in base; try 0.05) ────────────────────
run_exp "BU_ternary_noise"  TERNARY_NOISE_SCALE=0.05

# ── Grad clip norm ────────────────────────────────────────────────────────────
run_exp "BV_clip05"   GRAD_CLIP_NORM=0.5
run_exp "BW_clip01"   GRAD_CLIP_NORM=0.1

# ── Warmdown fraction (currently 0.5; try tighter schedule) ──────────────────
run_exp "BX_warmdown02"  WARMDOWN_FRACTION=0.2
run_exp "BY_warmdown08"  WARMDOWN_FRACTION=0.8

# ── Capsule dim 128 at 6L (biggest capsules) ─────────────────────────────────
run_exp "BZ_caps128"  SKC_CAPSULE_DIM=128 SKC_NUM_CAPSULES=8

# ── num_capsules sweep (8 fixed from earlier; try 4 and 16) ──────────────────
run_exp "CA_caps4"   SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=4
run_exp "CB_caps16"  SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=16

# ── BEST COMBO v2: incorporate all confirmed winners ──────────────────────────
# W=caps64, S=engram+lrlow, R=tko_off + any new winners from this batch
run_exp "CC_best_v2" \
    TKO_ENABLED=0 \
    BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3 \
    MATRIX_LR=0.02 SCALAR_LR=0.015 \
    SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=8 \
    MUON_MOMENTUM_WARMUP_STEPS=300 \
    NGRAM_CACHE_ENABLED=1

echo "=== BATCH-2 ABLATION COMPLETE: $(date) ==="
echo "BATCH2_COMPLETE: $(date)" >> missing_ablations_master.log
