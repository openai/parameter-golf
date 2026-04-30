#!/bin/bash
# Config A: iter-003.5 reproduction on 005.5 codebase
# N=4, seq=1024, LR=0.03, warmdown=1200, tied_emb=yes, stateful=no
cd /workspace/parameter-golf && mkdir -p logs

RUN_ID=cfg_a_repro \
MODEL_DIM=1536 N_BLOCKS=1 N_ITERS=4 \
D_STATE=32 EXPAND=2 HEADDIM=64 CHUNK_SIZE=64 \
MATRIX_LR=0.03 TIED_EMBED_LR=0.03 SCALAR_LR=0.03 \
TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 \
WARMDOWN_ITERS=1200 WARMUP_STEPS=5 LR_WARMUP_STEPS=200 \
GRAD_CLIP_NORM=1.0 \
TRAIN_STATEFUL=0 EVAL_STATEFUL=0 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=1 train_gpt_sg.py 2>&1 | tee logs/cfg_a_repro.txt
