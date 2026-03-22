# Parallel Muon + Parameter Banking + Polar Express

**Systems optimization: 82.14 ms/step (3.1% faster than PR #315's 84.76 ms/step)**

This is a pure training speed optimization built on top of [PR #315](https://github.com/openai/parameter-golf/pull/315) by @jfprincz (11L Partial RoPE + LN Scale + EMA + XSA4, val_bpb 1.1248). The model architecture and hyperparameters are unchanged — only the optimizer and weight storage layout are modified.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, 600s)

| | PR #315 Baseline | This Submission | Delta |
|--|---|---|---|
| **step_avg** | 84.76 ms | **82.14 ms** | **-2.62 ms (3.1%)** |
| **steps in 600s** | 7,079 | **7,306** | **+227 steps** |
| **val_bpb (pre-quant)** | 1.1421 | **1.1421** | identical |

> **Note on reported scores:** PR #315's headline val_bpb of **1.1248** is the post-quantization (int6+zstd) sliding window score. The pre-quantization val_bpb at wallclock cap is **1.1421** for both the baseline and this submission — the speed improvement is fully lossless. The artifact size is currently 20.4MB (above the 16MB track limit) due to the bank parameter layout affecting quantization grouping; this is a packaging issue, not a quality issue, and will be addressed in a follow-up.

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## What Changed (3 inseparable optimizations)

### 1. Parameter Banking
Replace 66 separate `nn.Linear` weight tensors with 4 contiguous 3D `nn.Parameter` banks:
- `qo_bank`: (22, 512, 512) — Q and Out projections for all 11 layers
- `kv_bank`: (22, 256, 512) — K and V projections
- `mlp_up_bank`: (11, 1536, 512) — MLP up projections
- `mlp_down_bank`: (11, 512, 1536) — MLP down projections

Forward pass uses `F.linear(x, bank[layer_idx])`. Verified: compiled forward+backward is identical speed (72.33ms vs 72.59ms on single GPU).

### 2. Polar Express ([arXiv:2505.16932](https://arxiv.org/abs/2505.16932))
Replace fixed Newton-Schulz coefficients (a=3.4445, b=−4.7750, c=2.0315) with per-iteration minimax-optimal polynomials. Same 5 iterations, same matmul structure, 35% tighter orthogonalization (0.21 vs 0.32 relative error in BF16).

### 3. Parallel Muon ([arXiv:2511.07464](https://arxiv.org/abs/2511.07464))
Remove DDP for bank parameters entirely. Handle gradient communication explicitly in the optimizer step with overlapped scheduling:

1. **Phase 1 (Scatter):** Launch async `reduce_scatter` for all banks (biggest first)
2. **Phase 2 (Adam):** `all_reduce` + step Adam on small params (scalars, embeddings) — overlaps with bank reduce-scatter in-flight
3. **Phase 3 (PE):** Wait for each bank's reduce-scatter, run local batched Polar Express on the GPU's shard (2-3 matrices per GPU instead of 22), launch async `all_gather` overlapped with next bank's PE

### Why DDP doesn't work with banking
Bank gradients aggregate across all 11 layers. A `qo_bank.grad` isn't "ready" until the very last layer's backward completes — DDP can't overlap its all-reduce with backward compute. This costs +4ms vs baseline. Removing DDP for banks and handling communication explicitly in the optimizer step restores full overlap.

## Credits

Built entirely on [PR #315](https://github.com/openai/parameter-golf/pull/315) by @jfprincz — the 11-layer architecture with Partial RoPE, LN Scale, EMA, and XSA4. This PR optimizes only the training speed; the model architecture, hyperparameters, initialization, and evaluation are unchanged.
