# 9. Joint Tokenizer and Model BPB Optimization

## Core Thesis

Because the score is bits per byte rather than token loss, the best tokenizer-model pair may not be the best token-level language model in the ordinary sense.

## What It Changes

Instead of taking the current tokenizer as fixed, jointly rethink:

- vocabulary size
- tokenization granularity
- model width/depth split
- the implied tokens-per-byte term in the final metric

## Why It Might Improve `val_bpb`

The challenge metric is tokenizer-agnostic on purpose. That means the winning system may exploit a better tokenizer/model trade than the default `sp1024` setup. In principle, a tokenizer that changes token-per-byte behavior favorably could win even if raw token CE does not look dramatically better.

## Why It Is Risky

This is easy to get wrong and will be examined closely. The current baseline already has more obvious bottlenecks, so tokenizer work is not where I would start.

## First Useful Experiment

Only do this after solving some optimization/export issues first. Then compare a small number of tokenizer variants with strict, careful bpb accounting.
