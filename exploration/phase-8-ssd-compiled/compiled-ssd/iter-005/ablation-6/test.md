# Ablation 6: Conv1d vs BCNorm

## Hypothesis

BCNorm (RMSNorm + learnable bias on B/C matrices, SiLU directly on x without conv1d) is causing the val_bpb regression from iter-003.5 (1.600) to iter-005.5 (1.98).

Standard Mamba-2 uses a short depthwise causal conv1d (kernel=d_conv) on x before the SSD scan. This provides local token smoothing -- a 1D causal convolution over the sequence dimension that lets each token see its d_conv-1 predecessors before entering the SSM. BCNorm replaces this with pointwise operations (RMSNorm + bias on B/C, SiLU on x), which have zero temporal receptive field.

The conv1d matters because:
- SSM dynamics are initialized from x at each position. Without smoothing, the SSM sees raw projected features that may be noisy/high-variance.
- Local n-gram patterns (bigrams, trigrams) are a large fraction of next-token-prediction signal at small scales. Conv1d captures these directly; BCNorm cannot.
- The conv1d acts as a low-pass filter that reduces aliasing before the SSM scan, analogous to anti-aliasing in signal processing.

BCNorm was introduced in Mamba-3 for large-scale models where the SSM has enough capacity to learn local patterns through the state. At our parameter budget (~14M params), the model may lack that capacity, making conv1d's inductive bias critical.

## What Changed

- Removed `bc_norm` (RMSNorm on B/C matrices)
- Removed `B_bias` and `C_bias` (learnable biases on B/C)
- Removed `B_bias`, `C_bias`, `bc_norm` from CONTROL_TENSOR_NAME_PATTERNS
- Added depthwise causal conv1d (kernel_size=d_conv, groups=d_inner) on x
- Added `conv1d` to CONTROL_TENSOR_NAME_PATTERNS (bias is a control tensor)
- x path: `conv1d(SiLU(x))` instead of just `SiLU(x)`
- B/C path: raw reshape (no normalization or bias) -- they still get RoPE

## Expected Outcome

If conv1d's local smoothing is the missing ingredient, this ablation should recover a significant portion of the BPB gap. The conv1d adds d_inner * d_conv + d_inner parameters (~8K for d_conv=4, d_inner=2048) but removes ngroups * d_state * 2 + small_norm parameters (~128 for ngroups=1, d_state=64). Net parameter change is negligible.

## How to Run

```bash
# 1xH100 smoke test (5 min)
RUN_ID=ablation_6_conv1d_vs_bcnorm \
MAX_WALLCLOCK_SECONDS=300 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Comparison Points

| Run | x processing | B/C processing | val_bpb | Notes |
|-----|-------------|----------------|---------|-------|
| iter-003.5 | SiLU (BCNorm) | RMSNorm + bias | 1.600 | Best result |
| iter-005.5 | SiLU (BCNorm) | RMSNorm + bias | 1.98 | Current code |
| ablation-6 | conv1d + SiLU | raw (no norm/bias) | ??? | This test |

## Interpretation Guide

- If ablation-6 val_bpb < 1.85: conv1d is a significant factor -- adopt it.
- If ablation-6 val_bpb ~= 1.98: conv1d is not the issue -- the regression is elsewhere.
- If ablation-6 val_bpb > 1.98: BCNorm was actually helping -- the regression source is something else entirely.
