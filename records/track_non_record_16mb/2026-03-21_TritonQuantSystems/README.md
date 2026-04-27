# Triton-Based Quantization Systems Approach

**Status: WIP — Active Development**

**val_bpb: TBD**

## Summary

Systems-level submission focused on custom Triton kernels for aggressive
sub-int8 quantization, enabling higher effective parameter density within
the 16MB artifact limit. Complementary to architecture-focused submissions
on the leaderboard.

## Approach

- Custom fused Triton kernels for quantized matmuls (mixed-bitwidth)
- Fused operator chains to reduce memory round-trips during training
- Mixed-precision packing with per-layer bitwidth selection
- Adopts proven community techniques (SWA, sliding window eval, Muon)

## Hardware

- Local iteration: RTX 3090 Ti (24GB)
- Final submission runs: 8xH100 (RunPod)

## Author

- GitHub: turbo-indubitable
- Name: Dan
