# 11L + XSA + Partial RoPE + LN Scale + EMA + Late QAT + TTT + TrigramHash + Adaptive Quant

## Summary

Combines all known SOTA techniques from PR #315 with additional improvements:

### Architecture (Phase 1)
- **11 layers** (up from 10) funded by mixed int5/int6 quantization savings
- **Exclusive Self Attention (XSA)** on last 4 layers: projects out self-value component from attention output via orthogonal projection, zero extra parameters
- **Partial RoPE** (16/64 dims): rotary embeddings on only 25% of head dimensions, remaining 75% use position-free attention for better long-range generalization
- **LN Scale damping**: each block's residual is scaled by `1/sqrt(layer_idx + 1)`, stabilizing deeper layers

### Training Dynamics (Phase 2)
- **EMA (decay=0.997)** instead of SWA: exponential moving average updated every step for smoother weight averaging and better post-quantization BPB
- **Late QAT with STE**: straight-through estimator fake-quantization enabled in the final ~4% of training (when lr_scale < 0.1), teaching the model to be robust to int6/int5 quantization noise
- **Batch size 524K** (down from 786K): yields ~22% more gradient steps in the 600s budget
- Muon optimizer with WD=0.04, momentum warmup 0.92->0.99 over 1500 steps, grad_clip=0.3

### Evaluation & Post-Training (Phase 3)
- **Backward-looking TTT**: 3 epochs of SGD (lr=0.002, momentum=0.9) on already-scored validation tokens, first 2 blocks frozen
- **Gradient-guided adaptive quantization**: accumulates squared gradient norms during final 10% of warmdown, assigns int7 to most sensitive tensors, int6 to medium, int5 to least sensitive MLPs
- **Sliding window eval stride=32** (down from 64) for more context coverage

### Additional (Phase 4)
- **TrigramHash embedding** (8192 buckets, dim=64): extends BigramHash to capture 3-gram token context
- **SmearGate** + **BigramHash** (10240 buckets, dim=128) + **OrthoInit** + zstd-22 compression

## Configuration

| Parameter | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 |
| MLP mult | 3.0x |
| RoPE dim | 16/64 |
| XSA layers | last 4 |
| Seq len | 2048 |
| Batch tokens | 524,288 |
| EMA decay | 0.997 |
| QAT threshold | lr_scale < 0.1 |
| TTT epochs | 3 |
| TTT lr | 0.002 |
| Eval stride | 32 |
| Warmdown iters | 3000 |

## How to Run

```bash
RUN_ID=11L_xsa_ttt SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_PartialRoPE_EMA_QAT_TTT/train_gpt.py
```
