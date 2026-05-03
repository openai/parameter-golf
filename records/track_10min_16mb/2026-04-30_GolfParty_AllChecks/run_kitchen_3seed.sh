#!/usr/bin/env bash
set -euo pipefail
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_ParamGolfKitchen_AllChecks
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
for SEED in 42 1234 0; do
    LOG=/tmp/kitchen_3seed_seed${SEED}.log
    echo === KITCHEN 3SEED SEED $SEED START === | tee $LOG
    date >> $LOG
    DATA_PATH=$DATA_PATH \
    TOKENIZER_PATH=$TOKENIZER_PATH \
    DATA_DIR=/workspace/parameter-golf/data \
    CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
    ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
    TTT_ENABLED=1 PHASED_TTT_ENABLED=1 \
    PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2500 \
    TTT_LORA_RANK=80 \
    TTT_MASK=no_qv TTT_Q_LORA=0 TTT_V_LORA=0 \
    TTT_LOCAL_LR_MULT=0.75 \
    EVAL_SEQ_LEN=3072 TTT_EVAL_SEQ_LEN=3072 \
    QK_GAIN_INIT=5.25 \
    MATRIX_LR=0.026 MIN_LR=0.1 EMBED_BITS=7 \
    MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
    MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 \
    GRAD_CLIP_NORM=0.3 \
    FUSED_CE_ENABLED=1 SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
    SPARSE_ATTN_GATE_ENABLED=1 \
    LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_GROUP_SIZE=64 \
    LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
    AWQ_LITE_ENABLED=1 ASYM_LOGIT_RESCALE=1 \
    GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 \
    COMPRESSOR=pergroup NCCL_NET=Socket \
    KS_UT_DEPTH=1 KS_LONG_CONTEXT=1 KS_E2E_TTT=0 \
    KS_SSM_LAST_K=1 KS_JEPA_WEIGHT=0.0 \
    KS_DIFFUSION_FRAC=0.05 KS_HNET_CHUNK=8 KS_MEGAKERNEL=1 \
    TTT_RLA_ENABLED=1 TTT_RLA_ORTHO=1 \
    RUN_ID=kitchen3seed_seed${SEED} SEED=${SEED} \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee -a $LOG
    echo === KITCHEN 3SEED SEED $SEED DONE === | tee -a $LOG
    date >> $LOG
done
echo === KITCHEN ALL 3 SEEDS DONE ===
