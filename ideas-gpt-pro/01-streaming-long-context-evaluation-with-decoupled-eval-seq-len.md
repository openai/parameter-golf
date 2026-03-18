# 1. Streaming Long-Context Evaluation With Decoupled `EVAL_SEQ_LEN`

## Category

Evaluation and test-time compute

## Why

The current `eval_val` hard-resets context every `TRAIN_SEQ_LEN`, even though the challenge allows evaluation at any sequence length. That throws away usable prefix information at every block boundary.

A streamed evaluator with a long causal memory is the cleanest "free" gain in this codebase. Once eval context exceeds train context, add NTK/Yarn-style RoPE scaling.

## Tradeoffs

- Speed: eval gets slower if done naively, but chunked KV-cache eval should keep it reasonable while leaving training unchanged
- Size: essentially neutral outside of a small code-byte increase
- Complexity/risk: low-moderate; the main hazard is exact causal accounting across rank boundaries

## Repo Fit

This fits cleanly inside `eval_val` plus a logits-returning forward path.
