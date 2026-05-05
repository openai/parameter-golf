# Experiment 034: SwiGLU + High LR (0.06) + Warmdown 3600

## Status: RUNNING on instance 0

## Config:
- SwiGLU(h=672) + MATRIX_LR=0.06 + SCALAR_LR=0.06 + TIED_EMBED_LR=0.03
- WARMDOWN_ITERS=3600
- FP16 embedding passthrough
- Combining SwiGLU's per-step quality with PR#42's high LR

## Hypothesis:
PR#42 tested SwiGLU with LR=0.04 and said it was slow. But they didn't try LR=0.06.
If the higher LR works with SwiGLU, we get the best of both worlds:
- SwiGLU's ~0.003 per-step BPB advantage over relu²
- PR#42's longer warmdown benefit
The question is whether SwiGLU tolerates the 3x higher LR.
