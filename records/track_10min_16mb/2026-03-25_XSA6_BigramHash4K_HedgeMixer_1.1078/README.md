# XSA6 + BigramHash4K + GatedAttn + ValueResidual on Hedge Mixer Stack

## Summary

Forked from [@agalimova's PR #720](https://github.com/openai/parameter-golf/pull/720) (XSA6 + BigramHash4K + HedgeMixer baseline at 1.1078).
Added two lightweight architectural modifications that significantly improved performance.

## Architectural Changes

1. **Gated Attention** (`attn_gate`): A per-head learned FP32 scalar (init=1.0) multiplied against the attention output, allowing the model to learn head-specific magnitudes.
2. **Value Residual** (`lambda_v`): A per-block learned FP32 scalar (init=0.0) that injects a fraction of the initial embedding `x0` directly into the residual stream.

Both scalars are registered in `CONTROL_TENSOR_NAME_PATTERNS` to remain in FP32 and bypass GPTQ quantization.

## Results

| Seed | val_bpb | val_loss | eval_time |
|------|---------|----------|-----------|
| 42   | 1.08778131 | 1.83666831 | 504s |
| 1337 | 1.09024766 | 1.84083264 | 503s |
| 2025 | 1.09090710 | 1.84194607 | 506s |
| **mean** | **1.08964536** | **1.83981567** | — |

Artifact: 14,917,177 bytes (14.9MB). All seeds evaluated under 600s.

## Evaluation

- `EVAL_STRIDE=64` (matches official baseline default)
- All runs completed in ~503–506s (under the 600s hard limit)
- Hardware: 8×H100 SXM 80GB
- Compression: zstandard (zstd level 22)
