#!/bin/bash
# XNOR-Net EGGROLL — gradient-free fine-tuning from pretrained checkpoint
export OMP_NUM_THREADS=1

# --- Data ---
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024

# --- Architecture (must match checkpoint) ---
export NUM_LAYERS=10
export MODEL_DIM=1024
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export EMBED_DIM=384
export ACTIVATION=signsq
export SMEAR=0

# --- XNOR ---
export XNOR_GROUP_SIZE=256
export BINARIZE_ACTIVATIONS=2
export USE_INT8_KERNEL=0
export USE_TRITON_KERNEL=0

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

# --- EGGROLL ---
export EGGROLL_LOAD=checkpoints_n2_100k/ckpt_step0100000.pt
export EGGROLL_LORA_RANK=4
export EGGROLL_LAYERS=0
export POP_SIZE=16384
export EGGROLL_SIGMA=0.01
export EGGROLL_LR=0.001
export EGGROLL_RANK=8
export FITNESS_SHAPING=rank
export LR_SCHEDULE=cosine
export WARMDOWN_FRACTION=0.3

# --- Schedule ---
export SEQ_LEN_SCHEDULE=0
export TRAIN_BATCH_TOKENS=65536
export TRAIN_SEQ_LEN=1024
export LR_WARMUP_STEPS=0
export MAX_WALLCLOCK_SECONDS=0
export ITERATIONS=50

# --- Eval ---
export VAL_LOSS_EVERY=10
export TRAIN_LOG_EVERY=1
export CHURN_LOG_EVERY=0
export VAL_MAX_TOKENS=0
export TEMP_SCALING=0
export SLIDING_EVAL=0
export SLIDING_EVAL_STRIDE=16
export SLIDING_BATCH_SIZE=512

# --- Checkpointing ---
export SEED=42
export COMPILE_MODE=default
export CHECKPOINT_EVERY=0

# --- Run ---
export RUN_ID=E11_eggroll-lora_8xH100_$(date +%s)
torchrun --standalone --nproc_per_node=8 train_gpt_cuda_xnor_eggroll_lora.py