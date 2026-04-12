# 12L sp9000 + Depth Recurrence + Hash-TTT — val_bpb 1.1036

**val_bpb: 1.1036** | **Artifact: 15.70 MB** | 8×H100 SXM 80GB

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1033 |
| Post-quant val_bpb | 1.1260 |
| Sliding window (stride=64) | 1.1138 |
| **TTT + hash embedding** | **1.1036** |
| Training steps | 4,292 |
| Step time | 139.8 ms |
| Total training time | 600s |
| TTT eval time | 233s |
| Artifact size | 15,700,910 bytes |
| Parameters | 36,098,144 |
| Seed | 1337 |

## What I Changed

This submission is built on the competition baseline with three main contributions:

1. **Custom sp9000 tokenizer** — I trained a 9000-vocab SentencePiece BPE tokenizer on the competition data, reducing token count by ~12% compared to sp1024 while maintaining identical byte-level accounting. Hosted at [Idan3011/parameter-golf-sp9000](https://huggingface.co/datasets/Idan3011/parameter-golf-sp9000).

2. **12-layer architecture with depth recurrence** — 12 physical layers with a 2-layer recurrence loop (layers 4–5 repeated 3×), giving 16 effective layers. To my knowledge, this is the first submission using 12 physical layers — most competitors use 10–11. The extra capacity is offset by the larger tokenizer reducing the number of tokens to process.

3. **Step-time optimization through code-level improvements** — I reduced step time from ~170ms to ~135ms purely through code optimization, without changing the model architecture or writing custom kernels. This added ~600 extra training steps, directly improving pre-quant quality.

4. **Improved TTT with bigram hash embedding** — I tuned the TTT hyperparameters (smaller chunks, higher LR, fewer epochs) and added a zero-initialized bigram hash embedding that learns token-pair patterns during eval-time adaptation. This improved TTT gain from −0.001 to −0.010.

## Speed Optimization Journey

The primary contribution of this submission is demonstrating that significant step-time improvements are achievable through careful code optimization alone, without custom CUDA/Triton kernels.

| Optimization | Step Time | Steps | Impact |
|---|---|---|---|
| Baseline (naive) | ~170 ms | ~3,530 | — |
| + `torch._foreach_*` weight decay (1 fused kernel vs 72 per-param) | ~163 ms | ~3,680 | −7 ms |
| + `torch._foreach_*` EMA update (2 kernels vs 226, no state_dict rebuild) | ~156 ms | ~3,846 | −7 ms |
| + RoPE precomputed once in `_run_blocks` (1× vs 17× recomputation) | ~151 ms | ~3,974 | −5 ms |
| + (B,S,H,D) attention layout (FA3-native, no transpose) | ~146 ms | ~4,110 | −5 ms |
| + `torch._foreach_*` Muon param update (1 kernel vs 72) | ~143 ms | ~4,196 | −3 ms |
| + SWA cached refs (no state_dict rebuild per checkpoint) | ~140 ms | ~4,286 | −3 ms |
| **Total** | **139.8 ms** | **4,292** | **−30 ms (−18%)** |

Each optimization targets a specific bottleneck identified through manual profiling. The foreach operations replace Python-level loops over individual parameters with fused CUDA kernels that operate on all tensors simultaneously. The RoPE precomputation eliminates redundant trigonometric calculations across the 16 effective layers. The (B,S,H,D) layout avoids transpose operations before FlashAttention 3.

## Architecture

| Component | Setting |
|-----------|---------|
| Physical layers | 12 |
| Effective layers | 16 (loop layers 4–5 × 3) |
| Model dim | 512 |
| Attention heads | 8 query, 4 KV (GQA) |
| MLP | 3.5× (hidden 1792), LeakyReLU(0.5)² |
| Tokenizer | sp9000 (custom 9000-vocab SentencePiece BPE) |
| Embeddings | Tied, SVD-scaled init (std=0.005) |
| XSA | All layers |
| Parallel residuals | Layers 7+ |
| Skip connections | U-Net style (encoder→decoder with learned gates) |
| Logit softcap | 30.0 |
| QK gain | 1.5 |
| RoPE base | 10000 |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | Muon (momentum=0.99, backend_steps=5, warmup from 0.92 over 1500 steps) |
| Optimizer (scalars/embeddings) | AdamW (fused, β1=0.9, β2=0.95) |
| Matrix LR | 0.022 |
| Scalar LR | 0.025 |
| Weight decay | 0.095 (Muon and Adam) |
| EMA decay | 0.9965 |
| SWA | Last 33% of training, every 5 steps, 50/50 blend with EMA |
| Warmdown | Wallclock-based, frac=0.72 |
| Batch tokens | 786,432 |
| Sequence length | 2048 |
| Warmup steps | 20 |

## Quantization

GPTQ with SD-Clip scaling:

| Parameter | Value |
|-----------|-------|
| Clip range | 31 (63 quantization levels) |
| Scale | k × std(row) / cr, k=15.0 |
| Calibration | 16 AR-generated sequences |
| Embedding quant | int8 per-row absmax |
| Compression | Brotli quality=11 + byte-shuffle |

Post-quant gap: +0.023 BPB. The gap is a function of depth — 16 effective layers compound quantization error through the recurrence loop. This is consistent across all quantization settings I tested (sweeping k from 12.85 to 20, sparsity 0–20%, per-layer absmax variants).

## TTT (Test-Time Training)

Legal score-first TTT following the PR #1413 pattern:

1. Val tokens split into 1,221 chunks of 32,768 tokens each
2. For each chunk:
   - **Score**: sliding window eval under `torch.no_grad()` — no gradients, no weight mutation
   - **Train**: SGD on the already-scored chunk (3 epochs, cosine LR)
3. Last chunk scored but never trained on
4. Chunk N is scored by a model adapted only on chunks 0..N−1

| TTT Parameter | Value |
|---------------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 0.01 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Gradient clip | 1.0 |
| Frozen blocks | None |

### Bigram Hash Embedding

I added a zero-initialized 16384×512 bigram hash embedding that is created fresh at eval time (not stored in the artifact). During TTT, it learns token-pair patterns at 10× the base learning rate:

```python
prev = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
hash_ids = (prev.long() * 2039 + input_ids.long()) % 16384
x = x + self.eval_hash_emb(hash_ids)
```

This gives the model a fast-learning dedicated memory for bigram patterns, separate from the slower-adapting base weights. The hash embedding absorbs local bigram statistics while the base model weights adapt to higher-order patterns.

| TTT Component | BPB | Gain |
|---|---|---|
| Post-quant (no sliding, no TTT) | 1.1260 | — |
| + Sliding window (stride=64) | 1.1138 | −0.012 |
| + TTT (old settings: 128K chunks, 20 epochs, lr=0.003) | ~1.1128 | −0.001 |
| + TTT (tuned: 32K chunks, 3 epochs, lr=0.01) + hash embedding | **1.1036** | **−0.010** |

### Timing Budget

| Phase | Time |
|-------|------|
| Training (wallclock cap) | 600s |
| Post-quant eval + sliding | ~145s |
| TTT (score-first + adaptation) | ~233s |
| **Total eval** | **~378s (< 600s budget)** |

## sp9000 Tokenizer

I trained a custom 9000-vocab SentencePiece BPE tokenizer on the competition's FineWeb data. Compared to the default sp1024:

- ~12% fewer tokens for the same text → more efficient training
- Larger embedding table (9000 vs 1024) → more artifact bytes for embeddings
- Net effect: positive, as the token reduction improves per-step quality more than the embedding overhead costs

The tokenizer and pre-tokenized data shards are hosted at [Idan3011/parameter-golf-sp9000](https://huggingface.co/datasets/Idan3011/parameter-golf-sp9000) (80 train shards + 1 val shard + tokenizer model).

## What I Tried That Did Not Work

| Experiment | Result | Why it failed |
|---|---|---|
| SOTA hyperparameters (momentum 0.97, grad_clip 0.3, ema 0.997) | Pre-quant +0.002 worse | Shrunk all weights 40% uniformly without helping quant gap. Tuned for longer runs (4500+ steps) than I achieve. |
| 18 effective layers (loop_start=3) | Quant gap +0.423 (catastrophic) | Error compounds through 3-block recurrence × 3 passes = 9 reuses of same quantized weights |
| Custom fused Triton MLP kernel (GEMM+act+sq) | 3ms slower on H100 | cuBLAS beats Triton for these GEMM shapes on Hopper. 1.5× faster on RTX 4050 but didn't transfer. |
| MLP 4.0× (wider MLP) | Artifact 16.88 MB (over cap) | Better pre-quant (1.1018) and smaller gap (+0.006) but model too large to fit |
| WD=0.50 on last block | Gap −0.002 only | Crushed block 11 weights 81% but raised kurtosis → harder to quantize |
| 128 GPTQ calibration sequences | Gap +0.007 worse | Broader Hessian + SD-Clip clips more tail rows on these weights |

## Previous Submission

[PR #1431](https://github.com/openai/parameter-golf/pull/1431) — 10L sp4096, val_bpb 1.1266. This submission improves on it by:

| | PR #1431 (old) | This submission |
|---|---|---|
| Architecture | 10L 3.5× sp4096 | 12L 3.5× sp9000 |
| Effective layers | 10 | 16 (depth recurrence) |
| Step time | 100.8 ms | 139.8 ms |
| Steps | 5,952 | 4,292 |
| Pre-quant | 1.1427 | 1.1033 |
| TTT gain | −0.001 | −0.010 |
| **Final val_bpb** | **1.1266** | **1.1036** |
| **Improvement** | — | **−0.023** |

## Credits

- **LeakyReLU² activation**: PR #493 by @parinzee
- **Score-first TTT**: PR #1413 by @dexhunter

