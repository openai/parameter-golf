# 15. Tiny Online Adaptation of Output Biases During Evaluation

## Category

Evaluation and test-time compute

## Why

Per-document or per-stream unigram shifts can help a lot on web text.

## Tradeoffs

- Speed: eval slows somewhat
- Size: tiny
- Complexity/risk: moderate-high

## Repo Fit

This is easy to wire in.

## Validity Risk

This is borderline. A lightweight causal cache is probably fine, but actual gradient updates on validation tokens are exactly the kind of move that could trigger "not in the spirit" pushback. It should be treated as a late-stage experiment, not a first-line submission strategy.
