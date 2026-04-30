# SOTA Architecture Decode (PR #1493 / 2026-04-09)

Decoded from `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` (LZMA-compressed, ~16.6KB on disk → ~48KB Python).

## Model

- **Vocab:** 8192 (SentencePiece, `sp8192` data variant)
- **Layers:** 11
- **Hidden:** model_dim=512, embedding_dim=512 (no projection)
- **Heads:** 8 query heads, 4 KV heads (GQA), head_dim=64
- **MLP:** 4× expansion, LeakyReLU(0.5)² activation (`fc → leaky_relu(0.5).square() → proj`)
- **RoPE:** partial, 16/64 dims rotated, base=10000, train_seq_len=2048
- **Attention scaling:** learnable per-head `q_gain` initialized to 5.0
- **LN:** RMSNorm (no learnable scale), with per-block multiplicative `attn_scale`/`mlp_scale` parameters and `ln_scale_factor = 1/sqrt(layer_idx+1)`
- **Tied embeddings:** yes; init std=0.005
- **Logit softcap:** 30 × tanh(logits/30)

## Block

```python
mix = resid_mix.to(dtype)            # [2, dim] — value residual mixing
x_in = mix[0] * x + mix[1] * x0      # x0 = post-embed signal carried through
attn_out = attn(attn_norm(x_in) * ln_scale_factor)
if parallel:                          # layers 7+
    mlp_out = mlp(mlp_norm(x_in) * ln_scale_factor)
    x_out = x_in + attn_scale * attn_out + mlp_scale * mlp_out
else:
    x_out = x_in + attn_scale * attn_out
    x_out = x_out + mlp_scale * mlp(mlp_norm(x_out) * ln_scale_factor)
```

- **Parallel residuals** from layer 7 onwards (GPT-J style — attn and MLP read same input, write to same output).
- **Sequential residuals** for layers 0–6.

## Depth recurrence

- `num_loops=2`, `loop_start=3`, `loop_end=5` — segment is `[3,4,5]`
- All-indices construction: `[0,1,2] + [3,4,5] + [3,4,5] + [3,4,5] + [6,7,8,9,10]` = 17 virtual layers
- Split at midpoint into encoder / decoder for U-Net skips
- **Skip connections** with learnable `skip_weights` (per-dim) and `skip_gates` (per-dim sigmoid for lerp)
- **Activation:** triggers at `enable_looping_at=0.35` (i.e. ~step 1592 of 4550)

## Attention (XSA on all 11 layers)

```python
y = flash_attn_3_func(q, k, v, causal=True)   # standard FA3
if use_xsa:
    # Subtract value-direction projection per KV-group
    y_g = y.reshape(B, T, Hkv, group, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)
    proj = (y_g * vn).sum(-1, keepdim=True) * vn
    y = (y_g - proj).reshape(B, T, H, D)
```

## Optimizers

- **Muon** (custom): row-normalized, 5 Newton-Schulz steps, momentum 0.99 (warmup 0.92→0.99 over 1500 steps), nesterov, weight decay 0.095. Applies to all 2D matrices in blocks **except** control tensors.
- **AdamW** (token embeddings): lr=0.03 (tied) or 0.6 (untied), wd=0.085
- **AdamW** (scalars + control tensors): lr=0.02, wd=0.02
- **Adam** (lm_head, only if untied): lr=0.008
- **Control tensor patterns** (excluded from Muon): `attn_scale, attn_scales, mlp_scale, mlp_scales, resid_mix, resid_mixes, q_gain, skip_weight, skip_weights, skip_gates`

## Training schedule

- **Iterations:** 20000 max, but capped at 600s wallclock (≈ 4550 steps actual)
- **Batch:** train_batch_tokens=786432, train_seq_len=2048
- **Warmup:** 20 steps
- **Warmdown:** 0.72 of training (linear to min_lr=0)
- **Grad clip:** 0.3 norm
- **EMA:** decay 0.9965 (applied during training, EMA weights used for eval)

## Quantization (GPTQ + SDClip + Brotli)

- **Bits:** matrices=int6, embeddings=int8
- **Clip:** `clip = k * std(row)`, k=12.85 for matrices, k=20.0 for embeddings
- **GPTQ:** Hessian-aware per-column rounding, block_size=128
- **Calibration:** 64 batches from training data
- **Compression:** byte-shuffle (stride=2) → Brotli-11
- **Reserve:** 12s of training budget reserved for GPTQ at end

## Eval

Three eval modes available, selected by env vars:

1. `eval_val` — single-pass cross entropy (baseline, fast)
2. `eval_val_sliding` — sliding window with `eval_stride=64` (already a win, used as the "before TTT" number)
3. `eval_val_ttt` — sliding window + chunk-based TTT (the SOTA result)

### TTT (the part we're attacking)

```python
ttt_params = list(model.parameters())   # ALL params, no filter
optimizer = SGD(ttt_params, lr=0.005, momentum=0.9)

for ci, windows in enumerate(chunk_windows):
    # 1. Score all windows in this chunk under no_grad → accumulates loss/tokens/bytes
    # 2. If not last chunk and ttt_epochs > 0:
    #    cos_lr = ttt_lr * 0.5 * (1 + cos(pi * ci / (num_chunks-1)))
    #    for ep in range(ttt_epochs):  # default 3
    #        for window in shuffled(windows):
    #            forward + backward
    #            for p in ttt_params: all_reduce(p.grad)
    #            clip_grad_norm_(1.0)
    #            optimizer.step()
```

- **Chunk size:** 32768 tokens (`ttt_chunk_tokens`)
- **Epochs per chunk:** 3
- **LR schedule:** cosine across chunks, no schedule within a chunk
- **Distributed:** params live on each rank, gradients all-reduced manually (since ttt_params are fp32 and not DDP-wrapped)
- **Score-first compliance:** scoring of chunk N happens *before* training on chunk N (legality requirement)

## Key tensors and parameter counts

To estimate after architectural changes:
- Token embedding: 8192 × 512 = 4.19M params (int8 → 4.19MB)
- 11 blocks × per-block:
  - Attention: q(512×512), k(512×256), v(512×256), proj(512×512) = 655K
  - MLP: fc(512×2048), proj(2048×512) = 2.10M
  - Scales/control: ~3K
  - Block subtotal: ~2.76M
- 11 blocks: ~30.3M
- Skip weights/gates: 2 × num_skips × 512 ≈ 8K
- **Total params: ~34.5M**
- At int6 + int8 embed + brotli: ~16MB artifact

## Parameters that are NOT quantized (TTT-relevant!)

These stay fp32 because they're scalar/vector control tensors, **not** in the GPTQ pipeline:

- `q_gain` per block: 8 floats (per attn) × 11 blocks = 88 floats
- `attn_scale` per block: 512 floats × 11 = 5632
- `mlp_scale` per block: 512 floats × 11 = 5632
- `resid_mix` per block: 2 × 512 × 11 = 11264
- `skip_weights`: num_skips × 512
- `skip_gates`: num_skips × 512
- `tok_emb` is int8 not int6, but still quantized

**Total non-quantized control surface:** ~25K floats. Tiny. **This is what selective-TTT can adapt without dequant/requant overhead.**
