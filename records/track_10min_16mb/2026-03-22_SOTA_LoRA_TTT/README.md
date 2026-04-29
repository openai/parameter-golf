# 10L Int5-MLP + BigramHash(10240) + LoRA TTT

**val_bpb:** TBD (mean of 3 seeds, after TTT eval on quantized model)
**Author:** ritikmahy5
**Date:** 2026-03-22

## Method

This submission combines the current SOTA training setup (thwu1's 10L Int5-MLP) with per-document LoRA test-time training (TTT) at evaluation time — an approach that has never been combined with the best training techniques.

### Training (unchanged from SOTA)

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- Mixed int5 (MLP) / int6 (attention) quantization + zstd-22
- BigramHash(10240, dim=128), SmearGate, U-Net skip connections
- Muon optimizer WD=0.04, SWA start_frac=0.4
- relu^2, MLP 3x (hidden=1536), orthogonal init with muP-scaled output projections
- seq_len=2048, batch=786K tokens

### Evaluation (new: LoRA TTT)

After training and quantization, we add per-document LoRA test-time training:

1. **Document isolation:** Process validation set document-by-document using BOS boundaries
2. **LoRA targets:** Rank-8 adapters on `c_q`, `c_v` (all layers) and `lm_head`
3. **Chunk strategy:** chunk_size=256 within eval_seq_len=2048 context windows
4. **Training:** Score chunk first (accumulate BPB), then train LoRA adapters on the chunk's loss
5. **Reset:** LoRA weights reset between documents (no inter-document leakage)
6. **Batching:** Documents sorted by length, batch_size=64 for GPU efficiency

### Why TTT Works

The SOTA uses ~155s for sliding window eval, but the 10-minute eval budget gives us 600s — leaving ~445s of unused compute. LoRA TTT exploits this free budget:

- On the naive baseline (1.2244), TTT gave -0.003 BPB improvement on top of sliding window stride
- On the stronger SOTA model (1.1428), per-document adaptation should be at least as effective
- LoRA parameters are tiny (~100K per batch) — negligible memory overhead
- No changes to the model artifact — TTT code is ~3KB of the 64KB script

### Key Differences from samacqua's LoRA TTT

| Aspect | samacqua (baseline) | Ours (SOTA) |
|--------|-------------------|------|
| Base model | Naive baseline (1.2244) | SOTA 10L Int5 (1.1428) |
| Base model features | No SmearGate/BigramHash | SmearGate + BigramHash(10240) |
| Eval seq_len | 1024 | 2048 |
| Quantized model | int8 + zlib | int5/int6 + zstd-22 |
| Configurable grad steps | 1 | 1+ (env var) |

## Hyperparameters

All TTT hyperparameters are configurable via environment variables:

| Param | Env Var | Default | Description |
|-------|---------|---------|-------------|
| LoRA rank | `TTT_LORA_RANK` | 8 | Rank of LoRA adapters |
| Learning rate | `TTT_LORA_LR` | 0.01 | Adam LR for LoRA params |
| Chunk size | `TTT_CHUNK_SIZE` | 256 | Tokens per TTT chunk |
| Batch size | `TTT_BATCH_SIZE` | 64 | Documents per batch |
| Grad steps | `TTT_GRAD_STEPS` | 1 | Adam steps per chunk |
| Enable TTT | `TTT_ENABLED` | 1 | Set to 0 to disable TTT |

## Results

TBD — awaiting H100 runs.

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgments

Built on thwu1's SOTA submission (10L Int5-MLP + BigramHash(10240) + SWA) and samacqua's LoRA TTT idea.
