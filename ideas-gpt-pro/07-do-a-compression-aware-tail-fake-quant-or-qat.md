# 7. Do a Compression-Aware Tail: Fake-Quant or QAT

## Category

Optimization and training changes

## Why

The leaderboard score is on the quantized round-trip model, not the bf16 training checkpoint. The current exporter is a post-hoc per-row or per-tensor int8 scheme with percentile clipping.

Simulating that exact path in the last `300` to `800` steps would optimize the weights for the artifact that actually gets submitted. It also opens the door to more aggressive 6-bit, 4-bit, or codebook export later.

## Tradeoffs

- Speed: slight slowdown late in training only
- Size: neutral or smaller if it enables harsher quantization
- Complexity/risk: moderate

## Repo Fit

This is a reasonable extension around the existing quantizer and exporter.
