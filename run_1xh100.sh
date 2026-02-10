#!/usr/bin/env bash
USE_FLASH_ATTN=0 torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/train_gpt_1xh100.py
