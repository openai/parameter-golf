# 3. Late QAT for the Submitted Model

## Core Thesis

The last part of training should optimize the quantized-roundtrip model you actually submit, not an fp32/bf16 model that gets damaged after training.

## What Bottleneck It Attacks

This also attacks the quantization gap, but from the training side instead of the exporter side. Right now quantization is entirely post-hoc:

- save state
- quantize
- zlib compress
- dequantize
- measure final score

Relevant code:

- export path begins around `train_gpt.py:1076`
- final scored roundtrip eval is `train_gpt.py:1096-1119`

## Why It Should Improve `val_bpb`

The model is currently trained to minimize loss in high precision, while the scored artifact is the int8 roundtrip model. The stronger the checkpoint gets, the more painful this mismatch appears to become, based on the 4-hour run. Late fake quantize-dequantize on the large matrices should encourage weights to land in regions that survive the exact export path better.

This is especially attractive because the challenge score is based on the submitted artifact, not the pre-quant model.

## Expected Effect

- Training speed: slightly slower during the late QAT window
- Evaluation speed: unchanged
- Compressed artifact size: unchanged

## Difficulty

4/5

## Rule-Risk

1/5

## Smallest Decisive Experiment

Only during the last `500` to `2000` steps:

- fake quantize-dequantize the main block matrices using the same scheme as export
- leave embeddings and tiny control tensors alone at first

Then compare final exact roundtrip `val_bpb` against the same training run without late QAT.

## Recommendation Bucket

Serious record attempt
