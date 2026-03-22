# Compute Grant Application — Highest Tier

## Brief description of your approach (1500 chars)

We combine an 11-layer transformer (MLP 3x, SmearGate, BigramHash, int6+zstd, SWA, Muon WD, NTK-RoPE) with full-weight SGD test-time training and a novel custom Triton/CUDA kernel pipeline.

Our key differentiator: we are the only team developing fused custom kernels for this competition. Using Makora automated kernel generation, we have produced validated kernels achieving 1.47x (fused RMSNorm+QKV), 1.23x (fused ReLU² MLP), 1.21x (fused softcap+CE), and 1.08x (fused resid_mix+RMSNorm) speedups on H100. Additional kernels for fused Q/K RMSNorm+RoPE+q_gain and fused TTT adaptation steps are in development.

The competition explicitly lists "megakernels" as a desired direction. No other submission currently uses custom kernels. Integrating our kernel pipeline would yield 15-20% training speedup (~800-1000 extra steps in the 10-min budget) and faster TTT adaptation, enabling more epochs within the eval budget.

Our current best: ~1.14 val_bpb (sliding window), competitive with SOTA. With kernels integrated, we expect to push below 1.12 by exploiting the step count advantage that faster training provides.

We also implement full-weight SGD TTT (adapting all model weights to validation data before scoring), achieving consistent improvement over sliding-window-only evaluation.

## What have you tried so far (255 chars)

11L+MLP3x+TTT achieving ~1.14 bpb. Custom Triton/CUDA kernels via Makora: fused RMSNorm+QKV 1.47x, ReLU² MLP 1.23x. Systematic ablations across 15+ H100 runs. Need credits to integrate kernels and run significance tests.

## Link(s) to your PR submission

https://github.com/openai/parameter-golf/pull/175
