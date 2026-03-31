# Parameter Golf Research Program — Storage Lane

## Objective
Improve final post-export performance by reducing compressed bytes and minimizing quantization/export damage.

## Primary Principle
The submission is judged on exported artifacts, not idealized full-precision weights.

## What We Know
- Strong model families are currently losing meaningful BPB after export.
- Some promising cores are close to competitive pre-quant but miss after post-quantization/export.

## Priority Order
1. Reduce quantization gap
2. Recover byte margin under the 16MB cap
3. Explore export-aware parameterization
4. Only then chase small raw-loss gains

## Preferred Directions
- FP16 tied embedding export
- Quantization-aware or export-aware calibration
- Weight-decay or regularization changes that improve export robustness
- Compression-friendly parameter sharing or scaling

## Avoid
- Large core architecture changes
- Anything that increases bytes without a very clear export advantage
- Changes that only improve pre-quant metrics

## Guidance
- One conceptual change per experiment
- Evaluate success by post-export BPB and bytes, not raw loss
- Treat small size wins with minimal BPB regression as acceptable
