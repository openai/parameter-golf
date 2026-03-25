# 10L + XSA(last 4) + EMA + TrigramHash + Int5MLP + BigramHash(10240)

**val_bpb: TBD** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

Three additions over the `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` SOTA (val_bpb=1.14276):

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (3 seeds)
NCCL_CUMEM_ENABLE=0 SEED=42 torchrun --nproc_per_node=8 --standalone records/track_10min_16mb/2026-03-22_10L_XSA4_EMA_TrigramHash/train_gpt.py
NCCL_CUMEM_ENABLE=0 SEED=1337 torchrun --nproc_per_node=8 --standalone records/track_10min_16mb/2026-03-22_10L_XSA4_EMA_TrigramHash/train_gpt.py
NCCL_CUMEM_ENABLE=0 SEED=7 torchrun --nproc_per_node=8 --standalone records/track_10min_16mb/2026-03-22_10L_XSA4_EMA_TrigramHash/train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed beyond NCCL fix.

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|----------------|-------|
| 42   | TBD     | TBD            | TBD   |
| 1337 | TBD     | TBD            | TBD   |
| 7    | TBD     | TBD            | TBD   |
| **Mean** | **TBD** | | |
| **Std**  | **TBD** | | |

## Novel Contributions

### 1. XSA — Exclusive Self Attention (arXiv:2603.09078)

Standard attention output `y_i` contains a component aligned with the token's own value `v_i`.
This "self-value bias" grows with layer depth and reduces the attention's ability to aggregate
genuinely cross-token information. XSA removes it via orthogonal projection (zero extra params):

```
z_i = y_i - (y_i · v̂_i) * v̂_i    where v̂_i = v_i / ||v_i||
```

GQA-aware implementation (avoids `repeat_interleave` memory allocation):
```python
gs = num_heads // num_kv_heads
Vn = F.normalize(v, dim=-1).unsqueeze(2)          # (B, Hkv, 1, S, Dh)
y_g = y.reshape(B, num_kv_heads, gs, S, head_dim)  # (B, Hkv, gs, S, Dh)
y_g = y_g - (y_g * Vn).sum(-1, keepdim=True) * Vn
y = y_g.reshape(B, num_heads, S, head_dim)
```

Applied to last 4 layers (XSA_LAST_N=4). ~2ms total overhead. ~0.003–0.007 BPB gain.

### 2. EMA — Exponential Moving Average (decay=0.997, replaces SWA)

Maintains a GPU-resident shadow copy of all model parameters, updated each optimizer step:
```
ema[t] = 0.997 * ema[t-1] + 0.003 * param[t]
```
Applied to model before final quantization. More continuous than SWA's discrete averaging —
no need to choose a start_frac; the shadow naturally tracks convergence across all of warmdown.

### 3. TrigramHash (1024 buckets, dim=64)

Hash `(prev2, prev1, curr)` token triples → learned embedding, added to input representation.
Provides 3-gram context at embedding layer, complementing BigramHash. Novel: not in any other
submission. Quantized as int6 (proj) + int8 (embedding table). Adds ~55KB to compressed artifact.

```python
out[..., 2:] = (9533 * t[..., :-2] + 97 * t[..., 1:-1] + t[..., 2:]) % 1024
```

## Architecture

- **10 layers**, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3× expansion (hidden=1536), relu² activation
- SmearGate + BigramHash(**10240**, dim=128) + **TrigramHash(1024, dim=64)** [NEW]
- **XSA on last 4 layers** [NEW], orthogonal init + muP output scaling
- U-Net skip connections (encoder=5, decoder=5), tied embeddings

## Quantization & Artifact

- **Int5 [-16,15]** for MLP weights
- **Int6 [-32,31]** for attention, bigram.proj, trigram.proj weights
- **Int8** for bigram.embed, trigram.embed tables
- **FP16** for tok_emb, blocks.9.attn.c_k (last layer key proj)
- 3% magnitude pruning, zstd-22 compression

## Training Hyperparameters

- Muon: matrix_lr=0.02, WD=0.04, momentum=0.99 (warmup 0.92→0.99 over 1500 steps)
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps, seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- **EMA decay=0.997** (replaces SWA)
- Sliding window eval: stride=64, XSA_LAST_N=4

## Comparison vs SOTA

| Technique | SOTA (1.14276) | This submission |
|-----------|---------------|-----------------|
| Layers | 10 | 10 (same) |
| BigramHash buckets | 10240 | **10240** (restored) |
| XSA | ❌ | ✅ last 4 layers |
| EMA | ❌ (uses SWA) | ✅ decay=0.997 |
| TrigramHash | ❌ | ✅ 1024 buckets |
| Int5 MLP + Int6 Attn | ✅ | ✅ |

Built on `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` (thwu1) + XSA from arXiv:2603.09078.
