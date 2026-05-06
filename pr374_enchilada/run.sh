#!/bin/bash
# PR374 Enchilada: 12L/2KV/2.75xMLP + train@1024 + EMA + earlier QAT + longer warmdown
#
# Changes from PR374 v38 (1.1246 BPB):
#   Shape:  12 layers (was 11), 2 KV heads (was 4), 2.75x MLP (was 3x)
#           ~same param budget, more depth, ~15% fewer FLOPS/step
#   Speed:  train_seq_len=1024 (was 2048), eval stays 2048
#           partial RoPE (16/64 dims) + NTK scaling handles extrapolation
#   Quality: EMA decay=0.997 stacked on Tight SWA
#           Late QAT at scale<0.15 (was 0.1) — more int6 adaptation steps
#           Warmdown 3500 (was 3000) — longer convergence tail
#           VE layers shifted to 10,11 for 12L model

set -euo pipefail

NUM_LAYERS=12 \
NUM_KV_HEADS=2 \
MLP_MULT=2.75 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=2048 \
WARMDOWN_ITERS=3500 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=10,11 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
ADAM_WD=0.04 \
MUON_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
torchrun --nproc_per_node=8 train_gpt.py
