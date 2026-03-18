# QAT + GradClip Conservative

## What changed from baseline

This submission modifies **only the training procedure** -- no architectural changes to the 9-layer, dim-512 transformer with ReLU^2 MLP.

Two changes:

1. **Quantization-Aware Training (QAT)** via straight-through estimator (STE), activated during the last 30% of wallclock time (configurable via `QAT_START_FRAC`).
2. **Gradient clipping** default changed from 0.0 (disabled) to 1.0.

## Why QAT helps

The baseline loses ~0.0072 BPB when post-training quantization converts fp32 weights to int8. QAT uses a straight-through estimator to simulate int8 per-row quantization during training: the forward pass sees quantized weights, but gradients flow through as if quantization didn't happen. This teaches the model to place its weights in regions that survive int8 rounding, recovering most of the quantization gap.

QAT is applied to:
- All `CastedLinear` layers (attention Q/K/V projections, output projections, MLP layers)
- The tied embedding weight when used as the lm_head projection

QAT activates at 70% of wallclock time by default, giving the model time to converge to good weights before fine-tuning for quantization robustness.

## Why gradient clipping helps

Gradient clipping at norm 1.0 prevents occasional large gradient spikes from destabilizing training, especially during the Muon optimizer's Newton-Schulz orthogonalization. This is a standard stabilization technique that costs almost nothing in compute.

## Risk level

This is the **lowest-risk variant** -- no model architecture changes, no new hyperparameters that affect model capacity, and both techniques are well-established in the literature. The only risk is that QAT adds a small amount of noise during the last 30% of training, but the STE formulation ensures gradients remain unbiased.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRAD_CLIP_NORM` | 1.0 | Max gradient norm (0 = disabled) |
| `QAT_START_FRAC` | 0.70 | Fraction of wallclock after which QAT activates |
