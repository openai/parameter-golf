#!/bin/bash
# Missing ablations — 5-min runs, strictly sequential (one at a time).
# Run from the experiment directory:
#   nohup bash run_missing_ablations.sh > missing_ablations_master.log 2>&1 &

set -e
cd "$(dirname "$0")"

BASE_ENV="
  ARCHITECTURE=skc
  NUM_LAYERS=4
  MODEL_DIM=128
  NUM_HEADS=4
  NUM_KV_HEADS=2
  SEED=42
  TRAIN_SEQ_LEN=256
  TRAIN_BATCH_TOKENS=8192
  MAX_WALLCLOCK_SECONDS=300
  CURRICULUM_ENABLED=1
  CURRICULUM_PHASE1_SEQ=64
  CURRICULUM_PHASE2_SEQ=128
  LAWA_ENABLED=1
  LAWA_K=5
  SWA_ENABLED=1
  SMEARGATE_ENABLED=1
  TKO_ENABLED=1
  SKC_CAPSULE_DIM=32
  SKC_NUM_CAPSULES=8
  FEEDBACK_ENABLED=0
  CAPSULE_ENABLED=0
  SLIDING_EVAL=1
  SLIDING_EVAL_STRIDE=32
  ITERATIONS=100000
  MATRIX_LR=0.035
  SCALAR_LR=0.025
  VRL_ENABLED=0
  BIGRAM_HASH_ENABLED=0
  TTT_ENABLED=0
  NGRAM_CACHE_ENABLED=0
  XSA_START_LAYER=999
  TEMP_SCALING=0
"

run_exp() {
    local RUN_ID="$1"
    shift
    echo "========================================"
    echo "START: $RUN_ID  $(date)"
    echo "========================================"
    env $(echo $BASE_ENV | tr '\n' ' ') \
        RUN_ID="$RUN_ID" \
        "$@" \
        bash run_mlx_reasoner.sh 2>&1 | tee "${RUN_ID}.log"
    echo "========================================"
    echo "DONE: $RUN_ID  $(date)"
    echo "========================================"
}

# ---------------------------------------------------------------------------
# 6-Layer experiments
# ---------------------------------------------------------------------------

run_exp R_6L_tko_off \
    NUM_LAYERS=6 TKO_ENABLED=0

run_exp S_6L_engram_lrlow \
    NUM_LAYERS=6 BIGRAM_HASH_ENABLED=1 MATRIX_LR=0.02 SCALAR_LR=0.015

run_exp T_6L_vrl_lndamp \
    NUM_LAYERS=6 VRL_ENABLED=1 LN_SCALE_DAMPING=1 XSA_START_LAYER=4

run_exp U_6L_conv2 \
    NUM_LAYERS=6 SKC_CONV_KERNEL=2

run_exp V_6L_conv8 \
    NUM_LAYERS=6 SKC_CONV_KERNEL=8

run_exp W_6L_caps64 \
    NUM_LAYERS=6 SKC_CAPSULE_DIM=64 SKC_NUM_CAPSULES=8

run_exp X_6L_inside_out \
    NUM_LAYERS=6 INSIDE_OUT_TRAINING=1

run_exp Y_6L_deq_fixed \
    NUM_LAYERS=6 DEQ_FEEDBACK=1 CAPSULE_ENABLED=1

run_exp Z_6L_best_combo \
    NUM_LAYERS=6 TKO_ENABLED=0 BIGRAM_HASH_ENABLED=1 MATRIX_LR=0.02 SCALAR_LR=0.015 \
    LN_SCALE_DAMPING=1 VRL_ENABLED=1 XSA_START_LAYER=4

# ---------------------------------------------------------------------------
# 4-Layer gap-fill experiments
# ---------------------------------------------------------------------------

run_exp AA_4L_tko_off_proper \
    TKO_ENABLED=0

run_exp AB_vrl_lndamp \
    VRL_ENABLED=1 LN_SCALE_DAMPING=1 XSA_START_LAYER=2

run_exp AC_feedback_fixed \
    FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 CAPSULE_ENABLED=1

run_exp AD_deq_fixed_4L \
    DEQ_FEEDBACK=1 CAPSULE_ENABLED=1

run_exp AE_engram_orders3 \
    BIGRAM_HASH_ENABLED=1 ENGRAM_NUM_ORDERS=3

run_exp AF_muon_warmup \
    NUM_LAYERS=6 MUON_MOMENTUM_WARMUP_STEPS=300

echo "ALL ABLATIONS COMPLETE  $(date)"
