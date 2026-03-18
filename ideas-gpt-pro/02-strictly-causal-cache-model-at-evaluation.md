# 2. Strictly Causal Cache Model at Evaluation

## Category

Evaluation and test-time compute

## Why

Small language models leave obvious gains on local repetition and document-specific vocabulary. A simple interpolation

`p = lambda * p_model + (1 - lambda) * p_cache`

with unigram, bigram, trigram counts or a neural cache over recent hidden states can move bpb more than many architecture tweaks.

This matches the challenge's openness to test-time compute and fits compression-competition style thinking well.

## Tradeoffs

- Speed: moderate eval slowdown only
- Size: almost free in model bytes; mostly code growth
- Complexity/risk: moderate-high because it needs a logits path, careful causal bookkeeping, and tuning of the interpolation rule

## Repo Fit

This is doable in the current script once forward can optionally return logits.
