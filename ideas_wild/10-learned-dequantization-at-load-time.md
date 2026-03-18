# 10. Learned Dequantization at Load Time

## Core Thesis

Instead of treating dequantization as a fixed arithmetic inverse of int8 scales, you could store a compact quantized base plus a tiny learned reconstruction mechanism that improves the effective recovered weights.

## What It Changes

The current exporter saves:

- int8 tensors
- scales
- a few passthrough tensors

A wilder alternative is to save:

- a more aggressively compressed base
- plus a tiny learned decoder or residual corrector applied at load time

## Why It Might Improve `val_bpb`

This makes decompression itself part of the model design rather than a transparent utility step. If the decoder is tiny enough, it could buy back more quality than it costs in bytes.

## Why It Is Risky

This is easy to overengineer. The decoder itself costs bytes and may not outperform simpler “keep the worst rows in fp16” schemes.

## First Useful Experiment

Do not redesign the full export stack. Start with one tensor family and test whether a very small learned residual corrector beats a plain mixed-precision outlier rescue at equal bytes.
