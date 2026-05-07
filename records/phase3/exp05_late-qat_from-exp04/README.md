# Late Quantization-Aware Training (QAT)

## Score: val_bpb = TBD

## Hypothesis

Applying fake quantization (quantize → dequantize with STE) only when `lr_scale < 0.1` (final ~4% of training) lets the model learn robust quantized configurations without corrupting early convergence. Community shows this closes the quantization gap from ~0.023 to ~0.007 BPB.

## Changes from exp04

- Added `qat_threshold=0.1` hyperparameter
- Before each forward pass during warmdown (when `lr_scale < threshold`): fake-quantize all 2D weight matrices using the same int5/int6 clip ranges as post-training quantization
- Uses `_classify_param` to match int5 (MLP) vs int6 (attn) clip ranges
- STE: gradient flows through the quantized weights unchanged

## Architecture

Inherits from exp04 (Partial RoPE + LN Scale + EMA + XSA) + Late QAT.

## Expected Impact

Reduce quantization penalty from ~0.02 to ~0.007 BPB.

## Results

TBD — awaiting A100 run.
