# 6. Train Shorter Than You Evaluate

## Category

Optimization and training changes

## Why

The baseline pays quadratic attention cost for 1024-token training from step 1, but evaluation can use longer context anyway.

A strong branch is `TRAIN_SEQ_LEN=512` with long streamed eval, or a `512 -> 1024` schedule late in training. That either buys more tokens per second or lets the run afford a larger model.

## Tradeoffs

- Speed: faster training early, slower eval
- Size: unchanged
- Complexity/risk: moderate because `torch.compile(dynamic=False)` dislikes shape changes

## Repo Fit

This still fits cleanly inside the current script as either a simple two-stage schedule or two precompiled shapes.
