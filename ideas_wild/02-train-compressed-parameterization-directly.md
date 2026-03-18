# 2. Train Compressed Parameterization Directly

## Core Thesis

Instead of training dense weights and compressing them afterward, make the compressed form itself the trainable object.

## What It Changes

The current pipeline is:

- train dense/high-precision weights
- quantize after training
- score the roundtripped model

A more challenge-native design would train:

- low-rank factors
- codebook indices and centroids
- blockwise quantized parameters
- or some other compressed latent that decodes into compute weights

Relevant baseline code:

- post-hoc quantization starts at `train_gpt.py:321`
- export and scoring path is `train_gpt.py:1076-1119`

## Why It Might Improve `val_bpb`

The current export gap is a real tax on the final score. If the model is trained in the parameterization that will actually be submitted, you stop paying that mismatch. This is one of the most challenge-specific directions available.

## Why It Is Risky

Optimization gets harder immediately. Many compressed parameterizations are awkward to train, and code complexity rises fast.

## First Useful Experiment

Do not redesign the whole model at once. Start by replacing only the biggest block matrices with a trainable low-rank form and leave the rest dense. Compare exact final artifact score, not only pre-export validation.
