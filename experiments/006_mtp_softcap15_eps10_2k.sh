#!/bin/bash
# Experiment 006: MTP (unnormalized) + softcap15 + eps1e-10 — COMPLETED, MTP HURTS
cd /home/ubuntu/parameter-golf
export WANDB_API_KEY=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
export WANDB_PROJECT=parameter-golf
export CUDA_VISIBLE_DEVICES=0
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export RUN_ID=exp006_mtp_softcap15_eps10
export LOGIT_SOFTCAP=15
export ADAM_EPS=1e-10
export MTP_ENABLED=1
python3 train_gpt.py
