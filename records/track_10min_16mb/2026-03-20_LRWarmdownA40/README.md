# LR warmdown on 1x A40 (non-record submission)

## what changed
- WARMDOWN_ITERS=3600
- MATRIX_LR=0.06

No architecture changes.

## results

- val_loss: 2.9096
- val_bpb: 1.7232
- total submission size: 8,397,395 bytes

## command

RUN_ID=cuda_lr_warmdown \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
MATRIX_LR=0.06 \
WARMDOWN_ITERS=3600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



