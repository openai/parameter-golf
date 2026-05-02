# Experiment 035: PR#42 Config + Speed Optimizations

## Status: RUNNING on instance 1 (relaunched with max-autotune after reduce-overhead CUDA graph crash)

## Config:
- Baseline relu² + MATRIX_LR=0.06 + WARMDOWN_ITERS=3600 + FP16 embed (PR#42)
- MUON_BACKEND_STEPS=3 (reduced from 5 — saves ~0.5-1ms/step)
- COMPILE_MODE=max-autotune (exhaustive kernel autotuning)
- enable_cudnn_sdp(True) (let PyTorch pick fastest attention backend)
- DDP: gradient_as_bucket_view=True, static_graph=True
- VAL_LOSS_EVERY=1000 (less validation overhead)
- BYTE_GROUPING=0 (disabled due to bug)

## Note: reduce-overhead (CUDA graphs) crashes with Rotary cache
The Rotary class caches cos/sin tensors that CUDA graphs then overwrite.
Would need to clone the cached tensors or use torch.compiler.cudagraph_mark_step_begin()
to fix. Using max-autotune instead which does exhaustive kernel search without CUDA graphs.

## Hypothesis:
max-autotune should find better kernel implementations for our matmuls.
MUON_BACKEND_STEPS=3 saves optimizer time. cuDNN SDP may beat Flash for GQA on H100.
Combined: expect 3-5% speedup = ~420ms/step vs 440ms baseline = ~500 more steps in 10 min.
