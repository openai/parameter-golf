#!/bin/bash
# XNOR-Net LLM

export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore::UserWarning:torch._inductor"

# --- Data ---
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024

# --- Architecture ---
export NUM_LAYERS=10
export MODEL_DIM=1024
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export EMBED_DIM=384
export ACTIVATION=signsq
export SMEAR=0
export ATTN_RES=2               # 0 = Standard U-Net with skip connections, 1 = AttnRes sub-layer, 2 = AttnRes pass-level

# --- XNOR ---
export XNOR_GROUP_SIZE=256
export BINARIZE_ACTIVATIONS=2   # 1 = full XNOR, 0 = weight-only binary (BWN), 2 = XNOR besides MLP
export USE_INT8_KERNEL=0
export USE_TRITON_KERNEL=1

# --- Attention ---
export ROPE_TYPE=yarn
export YARN_MAX_LEN=2048
export ROPE_BASE=5000
export QK_GAIN_INIT=2.25

# --- Logits ---
export LOGIT_SOFTCAP=10
export SOFTCAP_TYPE=poly
export TIE_EMBEDDINGS=1
export FP_STORAGE=FP8
export SCALE_STORAGE=BF16

# --- Optimizer ---
export MATRIX_OPTIMIZER=muon
export USE_GRAM_NS=0
export LR_SCHEDULE=cosine
export MATRIX_LR=0.008
export SCALAR_LR=0.01
export TIED_EMBED_LR=0.05
export HEAD_LR=0.02
export ADAM_LR=0.02
export ADAM_WD=0.04
export MUON_WD=0.04
export MUON_BACKEND_STEPS=3
export MUON_MOMENTUM=0.80
export MUON_MOMENTUM_WARMUP_START=0.80
export MUON_MOMENTUM_WARMUP_STEPS=200
export WARMDOWN_FRACTION=0.3
export SIGN_COMPRESS_REG=0.0
export GRAD_CLIP_NORM=1.0

# --- Schedule ---
export SEQ_LEN_SCHEDULE=1
export TRAIN_BATCH_TOKENS=65536
export TRAIN_SEQ_LEN=1024
export LR_WARMUP_STEPS=5
export MAX_WALLCLOCK_SECONDS=0
export ITERATIONS=100000

# --- Eval ---
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=1000
export CHURN_LOG_EVERY=0
export VAL_MAX_TOKENS=0
export TEMP_SCALING=1
export SLIDING_EVAL=1
export SLIDING_EVAL_STRIDE=16
export SLIDING_BATCH_SIZE=512

# --- EMA / Checkpointing ---
export EMA=0
export EMA_START_FRACTION=0.0
export EMA_DECAY=0.995
export SEED=42
export COMPILE_MODE=default
export CHECKPOINT_EVERY=10000

# --- Run ---
export RUN_ID=N3_full-xnor_8xH100_$(date +%s)
torchrun --standalone --nproc_per_node=8 train_gpt_cuda_xnor.py
