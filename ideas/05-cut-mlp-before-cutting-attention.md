# 5. Cut MLP Before Cutting Attention

## Core Thesis

If this baseline is over-allocating capacity anywhere, the `2x` ReLU squared MLP is the first place I would try to trim.

## What Bottleneck It Attacks

This attacks parameter allocation inside each block. The MLP is:

- `hidden = mlp_mult * dim`
- `fc: dim -> hidden`
- `proj: hidden -> dim`

Relevant code:

- `MLP_MULT`: `train_gpt.py:68`
- MLP definition: `train_gpt.py:608-617`

## Why It Should Improve `val_bpb`

Given the current layout, the MLP likely consumes more parameters per block than attention. In a tight byte budget regime, that is exactly the kind of component that can be too expensive relative to its marginal contribution. Shrinking or sharing MLPs is one of the easiest ways to free bytes for width, recurrence, or better quantization behavior.

I would not cut attention first because attention is already fairly compressed in this script:

- GQA is already used with `8` query heads and `4` KV heads
- tied embeddings are already enabled

That makes the MLP the more plausible over-allocation.

## Expected Effect

- Training speed: faster
- Evaluation speed: faster
- Compressed artifact size: smaller unless reinvested elsewhere

## Difficulty

3/5

## Rule-Risk

2/5

## Smallest Decisive Experiment

Compare:

- `MLP_MULT=1`
- current `MLP_MULT=2`
- one variant with smaller/shared MLP and larger `MODEL_DIM`

Evaluate only the best byte-matched configuration, not the raw smaller model.

## Recommendation Bucket

Serious record attempt
