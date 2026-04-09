# Non-record: Fused Triton Megakernels (RMSNorm + LeakyReLU²)

**val_bpb: 1.3560** | 1×RTX 5090, 600s | Beats baseline by 0.0017

## Summary

Custom Triton kernels for RMSNorm and LeakyReLU(0.75)² used during eval to speed up the evaluation phase, allowing slightly more training time within the wallclock budget. Training uses PyTorch with `fullgraph=True` torch.compile. Includes `torch.autograd.Function` wrappers for future training-time kernel use.

Implements one of OpenAI's requested research directions (Megakernels).

## Results (1×RTX 5090, 600s)

| Config | Steps | BPB |
|--------|-------|-----|
| Megakernel (Triton eval) | 975 | **1.3560** |
| Control (no Triton) | 958 | 1.3577 |

## Triton Kernels
1. `_rms_norm_fwd` — fused RMSNorm in single kernel pass
2. `_leaky_relu_sq_fwd` — fused LeakyReLU(slope)² avoiding intermediate write
3. PyTorch fallback when Triton unavailable or `MEGAKERNEL_ENABLED=0`
