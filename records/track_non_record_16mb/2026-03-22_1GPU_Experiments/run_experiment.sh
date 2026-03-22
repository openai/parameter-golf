#!/bin/bash
# RUN_1GPU_EXPERIMENT: Single H100 development/testing
# No DDP, no torchrun — avoids all the DDP crashes we hit on 8×H100
# Cost: ~$2-3/hr vs $21/hr for 8×H100
#
# Usage: bash run_1gpu_experiment.sh [experiment_name]
# Examples:
#   bash run_1gpu_experiment.sh baseline      # Proven config, 30 min
#   bash run_1gpu_experiment.sh backout       # Test BACKOUT=1, 30 min
#   bash run_1gpu_experiment.sh wd20k         # Test WD=20000, 30 min
#   bash run_1gpu_experiment.sh full_stack    # Everything, 2 hours
#   bash run_1gpu_experiment.sh moonshot      # GPTQ + Reptile + VE, 2 hours
#
# Results go to logs/<experiment_name>_<timestamp>.txt
# Compare val_bpb at same step count across experiments

set -e
cd /workspace/parameter-golf

EXPERIMENT=${1:-baseline}
echo "=== 1GPU Experiment: $EXPERIMENT ==="

# Common config (matches our 1.1375 proven config)
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1
export ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000
export EVAL_STRIDE=64 DOC_ISOLATED_EVAL=0 SEED=1337
export QAT=0 TTT_MAX_STEPS=500 TTT_FREEZE_BLOCKS=1
export TRAIN_BATCH_TOKENS=65536  # 1/8 of 524K for single GPU (same effective batch with grad_accum=8)

# Unset everything first
unset MLP_HIDDEN QUANT_BITS RUN_ID TIER2_MODE BIGRAM_HASH_BUCKETS \
  WARMDOWN_ITERS BACKOUT LAYER_DROP HEAD_DROP EVAL_TEMPERATURE \
  MLP_QUANT_BITS USE_FA3 LATE_K_FP16 EMA_ENABLED SWA PRUNE_PCT \
  GPTQ_LITE REPTILE_TTT VE_ENABLED

# 30-min default for quick experiments
export MAX_WALLCLOCK_SECONDS=1800
export VAL_LOSS_EVERY=500

case $EXPERIMENT in
  baseline)
    echo "Proven 1.1375 config — 30 min reference"
    git checkout int6-3xMLP-pr && git reset --hard origin/int6-3xMLP-pr
    ;;

  backout)
    echo "Test BACKOUT=1 — the -0.007 BPB we could never test on DDP"
    git checkout next-gen && git reset --hard origin/next-gen
    export BACKOUT=1
    ;;

  wd20k)
    echo "Test WD=20000 — smoother weights, better compression"
    git checkout int6-3xMLP-pr && git reset --hard origin/int6-3xMLP-pr
    export WARMDOWN_ITERS=20000
    ;;

  swa)
    echo "Test tight SWA instead of EMA — saves per-step overhead"
    git checkout int6-3xMLP-pr && git reset --hard origin/int6-3xMLP-pr
    export EMA_ENABLED=0 SWA=1
    ;;

  backout_wd20k)
    echo "BACKOUT + WD20K together"
    git checkout next-gen && git reset --hard origin/next-gen
    export BACKOUT=1 WARMDOWN_ITERS=20000
    ;;

  full_stack)
    echo "Everything proven — 2 hours"
    git checkout next-gen && git reset --hard origin/next-gen
    export MAX_WALLCLOCK_SECONDS=7200
    export BACKOUT=1 WARMDOWN_ITERS=20000
    export EMA_ENABLED=0 SWA=1
    export PRUNE_PCT=3.0
    ;;

  two_phase_ttt)
    echo "Two-phase TTT (norm-only + selective blocks) — the 1.12x technique"
    git checkout next-gen && git reset --hard origin/next-gen
    export TTT_TWO_PHASE=1
    export TTT_P1_EPOCHS=50 TTT_P1_LR=0.01
    export TTT_P2_EPOCHS=10 TTT_P2_LR=0.005 TTT_P2_UNFREEZE_BLOCKS=3
    export TTT_BATCH_SEQS=64
    export TTT_MAX_STEPS=9999
    export XSA_LAST_N=0
    export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
    ;;

  reptile_ttt)
    echo "Reptile + two-phase TTT — targeting 1.11x"
    git checkout next-gen && git reset --hard origin/next-gen
    export TTT_TWO_PHASE=1
    export TTT_P1_EPOCHS=50 TTT_P1_LR=0.01
    export TTT_P2_EPOCHS=10 TTT_P2_LR=0.005 TTT_P2_UNFREEZE_BLOCKS=3
    export TTT_BATCH_SEQS=64
    export TTT_MAX_STEPS=9999
    export REPTILE_TTT=1
    export XSA_LAST_N=0
    export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
    ;;

  ve)
    echo "Shared Value Embedding (layers 9,10) — only #374 has this"
    git checkout next-gen && git reset --hard origin/next-gen
    export VE_ENABLED=1
    ;;

  moonshot)
    echo "Everything: Reptile + VE + two-phase TTT + GPTQ-lite — 2 hours"
    git checkout next-gen && git reset --hard origin/next-gen
    export MAX_WALLCLOCK_SECONDS=7200
    export TTT_TWO_PHASE=1
    export TTT_P1_EPOCHS=50 TTT_P1_LR=0.01
    export TTT_P2_EPOCHS=10 TTT_P2_LR=0.005 TTT_P2_UNFREEZE_BLOCKS=3
    export TTT_BATCH_SEQS=64
    export TTT_MAX_STEPS=9999
    export REPTILE_TTT=1 VE_ENABLED=1 GPTQ_LITE=1
    export XSA_LAST_N=0
    export LATE_K_FP16=0 FP16_EMBED_EXPORT=0
    ;;

  *)
    echo "Unknown experiment: $EXPERIMENT"
    echo "Options: baseline, backout, wd20k, swa, backout_wd20k, two_phase_ttt, reptile_ttt, ve, full_stack, moonshot"
    exit 1
    ;;
esac

echo ""
echo "Config: BACKOUT=${BACKOUT:-0} WD=${WARMDOWN_ITERS:-3000} SWA=${SWA:-0} EMA=${EMA_ENABLED:-1}"
echo "        REPTILE=${REPTILE_TTT:-0} VE=${VE_ENABLED:-0} GPTQ=${GPTQ_LITE:-0}"
echo "        PRUNE=${PRUNE_PCT:-0} WALLCLOCK=${MAX_WALLCLOCK_SECONDS}s"
echo ""

# Single GPU — no torchrun, no DDP
python3 records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
