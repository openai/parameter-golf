# 12L Group-INT4 MLP + INT6 Attn + BigramHash(10240)

**val_bpb: 1.1462** | **STATUS: OVER 16MB BUDGET (17.5MB) — needs pruning fix**

## Approach

Built on the 10L Int5-MLP + BigramHash(10240) + SWA foundation from @thwu1's top submission. The core insight: INT5 quantization wastes 3 bits per byte — group INT4 packing (2 values per byte) cuts MLP storage by 47%, freeing enough space for 2 additional transformer layers.

## Novel Contributions

### 1. Group INT4 Quantization for MLP (novel)
Standard INT5-in-int8 stores one 5-bit value per 8-bit byte — 37.5% wasted. Our group INT4 scheme:
1. Divide weight columns into groups of 64
2. Compute per-group scale: `max(|group|) / 7.0` (stored as fp16)
3. Quantize to [-8, 7] (4-bit symmetric range)
4. Pack 2 INT4 values per byte via nibble packing (low=even, high=odd indices)

Per MLP weight (1536x512): **417,792 bytes** vs 789,504 bytes (INT5) — **47% smaller**.
Roundtrip MSE: 4.6e-6 on typical weights. Attention stays at INT6 for precision.

### 2. 12 Transformer Layers (enabled by INT4 savings)
The ~6-7MB freed by INT4 MLP quantization allows 12 layers instead of 10, adding 20% more model capacity. The model was previously capacity-limited (not data-limited), so deeper > wider.

- Encoder: 6 layers, Decoder: 6 layers
- U-Net skip connections between encoder/decoder halves
- ~5,495 training steps at 109ms/step (vs ~6,402 steps at 94ms/step for 10L)

### 3. Matched Training Config
- `train_batch_tokens`: 786,432 (50% larger batch for stable gradients)
- `warmdown_iters`: 3,000 (longer LR decay for final convergence)

## Results

| Config | val_bpb | Steps | Model Size | Status |
|--------|---------|-------|------------|--------|
| 10L INT5 (baseline) | 1.1482 | 6,402 | 16.2 MB | Valid |
| **12L INT4 (this)** | **1.1462** | 5,495 | **17.5 MB** | Over budget |
| Delta | **-0.0020** | -907 | +1.3 MB | — |

## Known Issue: Size Budget

Total submission: 17,497,295 bytes > 16,777,216 (16 MB limit).
Over by ~720 KB. INT4 packed bytes have higher entropy than INT5-in-int8,
so zstd compresses them less effectively despite fewer raw bytes.

**Planned fixes:**
- ~~Increase magnitude pruning from 3% to 6-8%~~ **DONE: increased to 8%**
- Reduce bigram_dim 128→96 or bigram_vocab_size 10240→8192
- Potentially drop to 11 layers if needed

## Architecture

- **12-layer** transformer, dim=512, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), ReLU^2 activation
- SmearGate + BigramHash(10240, dim=128)
- U-Net skip connections, tied embeddings
- Orthogonal init with muP-scaled projections
- Logit softcap 30.0, SWA (frac=0.4, every 50 steps)

## Quantization Breakdown

| Component | Method | Storage |
|-----------|--------|---------|
| MLP weights (12 layers) | **Group INT4 (gs=64)** | ~10.0 MB |
| Attention weights (12 layers) | INT6 (per-row) | ~4.7 MB |
| BigramHash embedding | INT6 | ~1.3 MB |
| Control params, norms | FP32/FP16 passthrough | ~0.2 MB |
| tok_emb, blocks.8.attn.c_k | FP16 passthrough | ~1.1 MB |
| + zstd-22 compression | — | **17.4 MB total** |
