# Approach D: TurboQuant-Guided Mixed Precision

**Status: Testing**

## Concept

Apply TurboQuant+ findings (ICLR 2026) to weight quantization:

1. **V compression is free** — V/O projection weights at int3 in middle layers
2. **All quality degradation comes from K** — Q/K projection weights at int5 (high precision)
3. **Boundary layers are sensitive** — first 2 + last 2 layers at int5 for all weights

## Bit Width Assignment

| Weight type | Boundary layers (0,1,9,10) | Middle layers (2-8) |
|-------------|---------------------------|---------------------|
| Q, K projections | int5 | int5 |
| V, O projections | int5 | **int3** |
| MLP weights | int5 | **int3** |

Effective average: ~4.2 bits/param (vs 5.0 for uniform int5)

## Expected Impact

- Smaller artifact → more headroom for bigger model or less pruning
- QAT-aligned: each CastedLinear uses its own clip range during training
- GPTQ-aware: Hessian-based quantization respects per-tensor bit widths

## Architecture

Same as Approach B (d=576, 33.6M params, MLP 3.5x) with mixed precision quantization.

## Rule Compliance

- GPTQ calibration within 600s training budget
- No TTT re-scoring
- Artifact < 16MB (asserted)
- Eval < 600s (asserted)
