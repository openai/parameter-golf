# Ablation 4: LR Warmup Impact

## Hypothesis

LR warmup (lr_warmup_steps=200) is a significant contributor to the val_bpb regression from iter-003.5 (1.600) to iter-005.5 (1.98).

With ~985 steps in a 5-minute 1xH100 smoke test, 200 warmup steps = 20% of total training spent at reduced LR (ramping from 0.1x to 1.0x). iter-003.5 had NO LR warmup -- it went straight to full LR from step 0 (only CUDA warmup for torch.compile, which is separate and doesn't affect LR).

In short wall-clock-bound training, every step at sub-optimal LR is wasted capacity. The model needs to learn as fast as possible from step 0.

## What Changed

- `lr_warmup_steps` default: 200 -> 0 (no LR warmup)
- Everything else identical to iter-005.5

## Expected Outcome

Removing LR warmup should recover some of the BPB gap between iter-003.5 and iter-005.5. The lr_mul function returns 1.0 immediately when lr_warmup_steps=0, so the model trains at full learning rate from step 0.

## How to Run

```bash
# 1xH100 smoke test (5 min)
RUN_ID=ablation_4_no_lr_warmup \
MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Comparison Points

| Run | lr_warmup_steps | val_bpb | Notes |
|-----|----------------|---------|-------|
| iter-003.5 | 0 | 1.600 | Best result, no LR warmup |
| iter-005.5 | 200 | 1.98 | Current code, 20% training at reduced LR |
| ablation-4 | 0 | ??? | This test |
