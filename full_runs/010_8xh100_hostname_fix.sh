#!/bin/bash
cd /home/ubuntu/parameter-golf
MYIP=$(hostname -I | awk '{print $1}')
export MASTER_ADDR=$MYIP
export MASTER_PORT=29501
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0,lo
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export RUN_ID=exp010b_fast_softcap15
export VAL_LOSS_EVERY=200
export TRAIN_LOG_EVERY=200
torchrun --nnodes=1 --nproc-per-node=8 --rdzv-backend=static \
  --rdzv-endpoint=${MYIP}:29501 --master-addr=${MYIP} --master-port=29501 \
  train_gpt.py
