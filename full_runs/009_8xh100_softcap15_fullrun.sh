#!/bin/bash
# Experiment 009: Full 8xH100 production run with hostname fix
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export RUN_ID=exp009_8xh100_softcap15
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export VAL_LOSS_EVERY=200
export TRAIN_LOG_EVERY=200

# Fix hostname resolution for Thunder Compute
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=eth0,lo

# Use static rendezvous backend to avoid hostname lookups
torchrun \
  --nnodes=1 \
  --nproc-per-node=8 \
  --rdzv-backend=static \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --master-addr=${MASTER_ADDR} \
  --master-port=${MASTER_PORT} \
  train_gpt.py
