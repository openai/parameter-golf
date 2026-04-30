#!/bin/bash
# EXP-048: SP4096 + Full SOTA + TUNED HYPERPARAMETERS
# Applies Gemini+Codex joint recommendations:
#  - MUON_MOMENTUM 0.99 → 0.95 (less aggressive, both AIs agree)
#  - QK_GAIN_INIT 5.25 → 4.0 (Codex: too aggressive with partial RoPE)
#  - MATRIX_LR 0.022 → 0.02 (Gemini: high leverage)
# Baseline: EXP-042 val_bpb=1.5596
cd /Users/lucaslt/Documents/side-gig/openai/parameter_golf/repo
source .venv/bin/activate
export RUN_ID=exp048_tuned_hparams
export DATA_PATH=./data/datasets/fineweb10B_sp4096
export TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
export VOCAB_SIZE=4096
export NUM_LAYERS=11
export MLP_MULT=4
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export ITERATIONS=5000
export WARMDOWN_ITERS=1000
export WARMDOWN_KIND=cosine
# TUNED values (was 0.022, 0.99, 5.25)
export MATRIX_LR=0.02
export TIED_EMBED_LR=0.03
export SCALAR_LR=0.02
export MUON_MOMENTUM=0.95
export MUON_MOMENTUM_WARMUP_START=0.90
export MUON_MOMENTUM_WARMUP_STEPS=60
export GRAD_CLIP_NORM=0.3
export QK_GAIN_INIT=4.0
export TRAIN_BATCH_TOKENS=8192
export VAL_LOSS_EVERY=0
export VAL_BATCH_SIZE=524288
export TRAIN_LOG_EVERY=200
export MLX_MAX_MICROBATCH_TOKENS=4096
export GRAD_ACCUM_STEPS=2
export WARMUP_STEPS=10
export MAX_WALLCLOCK_SECONDS=0
# SOTA features (same as EXP-042 baseline)
export RECUR_LAYERS=3,4,5
export RECUR_START_STEP=0
export PARALLEL_RESIDUAL=1
export PARALLEL_START_LAYER=7
python3 train_gpt_mlx.py
