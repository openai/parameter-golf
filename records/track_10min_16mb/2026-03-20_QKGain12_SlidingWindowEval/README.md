# QK Gain Init 1.2 + Sliding Window Eval

Two orthogonal improvements over the naive baseline:

## Changes

### 1. QK Gain Initialization (training)
- `QK_GAIN_INIT=1.2` (default: 1.5)
- Lower gain initialization improves attention stability during short training runs
- On RTX 4090 (340 steps): int8+zlib bpb 1.6133 vs baseline 1.6353

### 2. Sliding Window Evaluation (eval-only)
- `EVAL_STRIDE=64`, `EVAL_BATCH_SEQS=32`
- Each token scored with 960+ tokens of context instead of 0-1023
- Added `forward_logits()` method and `eval_val_sliding()` function
- Free ~0.03 bpb improvement with no training changes

## Local Results (RTX 4090, ~340 steps)
- Baseline: 1.6353 int8+zlib bpb
- QK Gain 1.2 only: 1.6133 int8+zlib bpb
- Sliding window only: 1.5879 raw bpb

Expecting much better absolute numbers on 8xH100 with 10k+ steps.
