# 14. Sweep Tokenizer Size Upward a Bit

## Category

Tokenizer and data-interface changes

## Why

Tokenizer-agnostic scoring does not mean tokenizer-irrelevant scoring. A `2k` or `4k` SentencePiece vocab may improve the bpb frontier enough to be worth the modest extra embedding and head cost, especially once the setup stops optimizing around `1024` vocab as if embedding bytes dominated.

## Tradeoffs

- Speed: similar
- Size: slightly to moderately larger
- Complexity/risk: high because data must be retokenized and `val_bpb` must still be computed exactly

## Repo Fit

This is the least "inside `train_gpt.py`" of the serious ideas, and the README explicitly says tokenizer changes will be audited hard.
