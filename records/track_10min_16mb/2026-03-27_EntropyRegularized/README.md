# Entropy-Regularized QAT + Legal TTT (WIP - pending compute)

Built on the merged 1.1233 record (PR #374) with techniques from the 1.1194 record (PR #549).

## Changes from baseline

**Training:**
- LeakyReLU(0.5) activation (proven -0.003 BPB from PR #493)
- Differentiable entropy penalty on int6 quantized MLP weight symbols during late QAT window
- Entropy penalty gated by ENTROPY_LAMBDA env var (default 0, no behavior change)

**Eval:**
- Legal score-first TTT from PR #461/#549 (gated by TTT_ENABLED, default off)
- Sliding window eval stride=64

## Run command
ENTROPY_LAMBDA=0.003 TTT_ENABLED=1 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

## Status
Code complete, tested locally. Awaiting 8H100 runs for results.
