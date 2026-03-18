# 1. Recurrent Transformer With Few Unique Blocks

## Core Thesis

Under a hard 16 MB artifact cap, weight sharing is probably the single biggest structural lever still mostly unused by the current baseline.

## What It Changes

The current model instantiates each block separately in a `ModuleList`. A wilder alternative is:

- only `2` to `4` unique blocks
- unroll them many times
- add per-step gates, scales, or embeddings so repeated applications are not identical

Relevant baseline code:

- block stack definition: `train_gpt.py:674`
- encoder/decoder pass structure: `train_gpt.py:707-713`

## Why It Might Improve `val_bpb`

This challenge rewards compute-per-byte much more than conventional model design does. A recurrent/shared-weight stack lets you buy effective depth almost for free in artifact bytes. That creates room to reinvest bytes into width, improved embeddings, or export robustness while keeping runtime roughly similar.

It is also one of the few ideas that directly matches the README’s own examples of promising directions.

## Why It Is Risky

Shared-weight models can be harder to optimize and can collapse into ineffective repetition if they do not get enough per-step modulation.

## First Useful Experiment

Keep the current 9-step unrolled shape, but replace the 9 unique blocks with:

- 3 unique blocks reused 3 times each
- per-step scale vectors
- optional step embeddings added to the hidden state

Then widen the model until the artifact size approaches the current budget again.
