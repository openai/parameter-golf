# 2. Use Byte Slack for Quantization Outliers

## Core Thesis

The exporter should spend the remaining artifact slack on the worst quantization outliers instead of forcing nearly every large float tensor through the same int8 path.

## What Bottleneck It Attacks

This attacks post-training quantization damage. The current exporter uses:

- one global clip percentile
- per-row int8 for 2D tensors
- per-tensor int8 for vectors/scalars
- a coarse keep-float threshold

Relevant code:

- `INT8_KEEP_FLOAT_MAX_NUMEL=65536`: `train_gpt.py:304`
- `INT8_CLIP_PERCENTILE=99.99984`: `train_gpt.py:307`
- main quantization path: `train_gpt.py:321-399`

## Why It Should Improve `val_bpb`

The baseline 10-minute run already loses score after export:

- pre-quant at stop: `1.2172`
- post-roundtrip exact: `1.22436570`

The 4-hour run loses even more:

- pre-quant at stop: `1.1749`
- post-roundtrip exact: `1.20737944`

That gap is large enough to be a first-class optimization target. The baseline artifact total is `15,863,489` bytes, which leaves only `136,511` bytes of slack, but that is still enough to rescue a small number of bad rows or tensors. A mixed scheme that keeps the worst rows in fp16, or stores fp16 residuals for a limited set of rows, should buy back measurable loss for a tiny byte cost.

## Expected Effect

- Training speed: unchanged
- Evaluation speed: unchanged
- Compressed artifact size: slightly larger, but controllable to remain under the cap

## Difficulty

3/5

## Rule-Risk

1/5

## Smallest Decisive Experiment

Take one finished checkpoint and rank tensor rows by quantization error. Then:

- keep the top-N worst rows in fp16
- or store fp16 residuals for the worst rows
- stop when total artifact size gets close to but stays under `16,000,000`

Compare exact roundtrip `val_bpb` to the current exporter.

## Recommendation Bucket

Baseline script improvement
