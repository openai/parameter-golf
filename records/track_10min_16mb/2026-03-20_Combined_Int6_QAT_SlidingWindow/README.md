# Combined Int6 + QAT + Sliding Window

## Strategy
Combine the best techniques from the top two submissions plus novel QAT.

## Techniques from WarmdownQuantization (#1, 1.1574 bpb)
- **Int6 quantization** ([-31,31] range, better zlib compression than int8)
- **FP16 tied embeddings** (avoid int8 compounding on shared input/output matrix)
- **Late-K passthrough** (last 2 layers' key weights in fp16)
- **Aggressive warmdown** (WARMDOWN_ITERS=20000, entire training is LR decay)
- **Higher LRs** (MATRIX_LR=0.06, SCALAR_LR=0.06)
- **Grad clipping** (GRAD_CLIP_NORM=1.0)
- **9 layers, 3x MLP** (hidden=1536)

## Techniques from SlidingWindow (#2, 1.1748 bpb)
- **Batched sliding window eval** (stride=64, compiled forward_logits, batch_size=256)
- **Overtone spectral embedding init** (SVD power-law spectrum shaping)
- **Phase-transition resid_mix init** (sigmoid-scheduled)
- **Muon decoupled weight decay** (0.02 * lr after each step)
- **AdamW** for embeddings and scalar params (weight_decay=0.01)
- **Higher tied embed LR** (0.10 vs 0.07)

## Novel Contributions
1. **Quantization-Aware Training (QAT)** with straight-through estimator (STE):
   - In the last 30% of training, inject int6 quantization simulation in forward pass
   - Model learns to be robust to int6 quantization, reducing post-quant penalty to near-zero
   - Only applied to large weight matrices (>65K params), not small control tensors
2. **Cosine warmdown** instead of linear (smoother LR decay, better final weights)
3. **Higher Muon momentum warmup** (700 steps vs 500) for stability with higher LRs

## Reproduction
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
All hyperparameters are set as defaults in the script. Override via environment variables if needed.

## Expected Results
Target: ~1.150-1.155 bpb (improvement over 1.1574 baseline)
