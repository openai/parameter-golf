# 10L XSA + LeakyReLU² + Partial RoPE + Higher LR

**val_bpb: 1.1370** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

```bash
# Default seed=42
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With specific seed
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.13815 | 15,983,322 | yes |
| 1337 | 1.13601 | 15,968,675 | yes |
| 2024 | 1.13697 | 15,650,120 | yes |
| **Mean** | **1.13704** | | |
| **Std** | **0.00088** | | |

## Key Techniques (new vs SOTA baseline)

### 1. XSA (Exclusive Self Attention) — Last 4 Layers
Removes self-value projection from attention output, forcing the attention mechanism
to model only contextual information orthogonal to the self-representation. The token's
own information already passes through the residual stream.

```python
vn = F.normalize(v, dim=-1)
y = y - (y * vn).sum(dim=-1, keepdim=True) * vn
```

Source: arxiv:2603.09078. Used in all top pending PRs (#315, #349, #401).

### 2. Leaky ReLU² Activation (negative_slope=0.5)
Replaces relu² with leaky_relu(0.5)². Allows gradient flow through negative
activations while maintaining the squared nonlinearity. ~0.003 bpb improvement
validated by vukrosic lab across 60+ experiments.

### 3. Partial RoPE (16 of 64 dims)
Apply rotary position embeddings to only 25% of head dimensions. The remaining
75% attend without positional bias, acting as position-independent feature detectors.
Zero parameter overhead. Used in PRs #315 and #401 (1.1248 and 1.1243).

### 4. Higher Learning Rates
- matrix_lr: 0.02 → 0.025
- scalar_lr: 0.02 → 0.025
- tied_embed_lr: 0.03 → 0.035

All top pending PRs (#315, #349, #401) use 0.025. The single biggest improvement
found in our experiments.

## Architecture (unchanged from SOTA baseline)

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), leaky_relu(0.5)² activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training Hyperparameters

- Muon optimizer: matrix_lr=0.025, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 5% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- Sliding window eval: stride=64

## Quantization

- Int5 [-16,15] for MLP weights (clip_range=15)
- Int6 [-32,31] for attention weights (clip_range=31)
- FP16 for tied embeddings
- zstd-22 compression

Built on SOTA baseline by @thwu1 (PR #180).
