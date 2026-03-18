# 4. Maintain an EMA and Export the Averaged Weights

## Category

Optimization and training changes

## Why

Ten-minute runs are noisy, and EMA often improves validation more than almost any single hyperparameter tweak. It also tends to reduce outliers, which can slightly help compression and quantization.

## Tradeoffs

- Speed: negligible overhead
- Size: no submission-size increase if only the EMA copy is saved
- Complexity/risk: very low

## Repo Fit

This is one of the cleanest wins available.
