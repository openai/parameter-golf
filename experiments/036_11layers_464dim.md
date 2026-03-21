# Experiment 036: 11 Layers @ dim=464 — More Depth at Same Speed

## Status: RUNNING on instance 2 (relaunched with max-autotune)

## Config:
- NUM_LAYERS=11, MODEL_DIM=464 (vs baseline 9 layers, dim=512)
- Baseline relu² + MATRIX_LR=0.06 + WARMDOWN_ITERS=3600 + FP16 embed
- MUON_BACKEND_STEPS=3
- COMPILE_MODE=max-autotune
- Same total params (~17M) as baseline

## Hypothesis:
From thread 2 speed analysis: 11@464 has nearly identical step time as 9@512
but 22% more depth. If additional layers provide diminishing but nonzero value,
this is a FREE improvement. The narrower dim (464 vs 512) reduces per-layer
capacity but more layers may compensate.

Risk: dim=464 is not divisible by num_heads=8 (464/8=58 head_dim).
head_dim=58 is even (good for RoPE) but not power-of-2 aligned.
May need NUM_HEADS=4 (head_dim=116) for better tensor core alignment.

## Parameter Budget:
- Embedding: 1024 × 464 = 475K
- Per block: Q(464×464) + K(232×464) + V(232×464) + O(464×464) + MLP_fc(928×464) + MLP_proj(464×928)
  = ~1.5M per block
- 11 blocks: ~16.5M + embed = ~17M — fits 16MB ✅
