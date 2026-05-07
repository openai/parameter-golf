# GPTQ Second-Order Quantization

## Score: val_bpb = TBD

## Hypothesis

GPTQ uses Hessian information (X^T X from calibration data) to minimize quantization error via column-by-column second-order optimization. Community shows full GPTQ at 1.1154 vs lite at 1.1228 — ~0.007 BPB improvement over naive quantization.

## Changes from exp05

- Added `gptq_quantize_layer()`: Cholesky-based GPTQ with per-row scaling
- Hessian collection via forward hooks on calibration data (8 batches)
- `mixed_quantize_int6` now accepts `gptq_hessians` dict — uses GPTQ when Hessian available, falls back to naive otherwise
- AWQ + GPTQ stacked (AWQ scales columns before GPTQ quantizes)

## Architecture

Inherits from exp05 (Partial RoPE + LN Scale + EMA + XSA + Late QAT) + GPTQ post-training.

## Expected Impact

~0.007 BPB improvement in quantized model quality.

## Results

TBD — awaiting A100 run.
