#!/usr/bin/env bash
torchrun --standalone --nproc_per_node=8 train_gpt.py --config configs/train_gpt_8xh100.py
