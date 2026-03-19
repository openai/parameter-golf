# Submission: Optimal LR Warmdown Scheduling

**Author:** Majdi Zamim (@MajdiZamim)
**Score:** 1.2381 BPB
**Hardware:** 8×H100 SXM, 10 minutes

## Key Insight

The default `WARMDOWN_ITERS=1200` never fires when training is wallclock-capped at 600 seconds with slower hardware, causing the model to terminate without proper LR decay.

By setting `WARMDOWN_ITERS=3000`, the cosine warmdown fires in the final ~30% of training steps, producing significantly better convergence.

## Command
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=submission_v3 \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
WARMDOWN_ITERS=3000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Metric | Value |
|--------|-------|
| val_bpb (post-quant) | 1.2381 |
| val_loss | 2.0904 |
| steps | 9,618 |
| tokens | ~5B |
| artifact size | 15.85MB |
