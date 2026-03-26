# AnAI — Parameter Golf Competition Entry

**Target: val_bpb < 1.1378** (beating SOTA 1.1428 by ≥0.005)

## Run Command

```bash
# Setup (once per machine)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + evaluate on 8xH100
RUN_ID=anai_run1 SEED=42 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/AnAI/train_gpt.py

# With different seeds for statistical significance
RUN_ID=anai_run2 SEED=1337 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/AnAI/train_gpt.py
RUN_ID=anai_run3 SEED=2024 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/AnAI/train_gpt.py
```

## Architecture

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu² activation
- SmearGate + BigramHash(12288, dim=128) + **TrigramHash(4096, dim=64)**
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings (FP16)

## Key Techniques

### 1. Mixed Int4/Int5/Int6 Quantization (Novel)
- **Int4 [-8,7]** for MLP fc (upward projection) weights — most compressible, higher zstd ratio
- **Int5 [-16,15]** for MLP proj (downward projection) weights
- **Int6 [-32,31]** for attention weights (precision-sensitive)
- **FP16** for tied embeddings and last-layer key projections

The key insight: MLP upward projections (fc) are less precision-sensitive than downward projections (proj), so we can push them to int4 without significant quality loss. This frees ~2MB of artifact budget compared to uniform int5, which we reinvest in larger n-gram hash tables.

### 2. BigramHash(12288) — Increased from SOTA (10240)
Hash consecutive token pairs into 12288-bucket embedding table (dim=128), projected to model_dim=512. Each bucket increase monotonically reduces hash collisions. The ablation from SOTA showed:
- bigram=4096: baseline
- bigram=8192: -0.0012 bpb
- bigram=10240: -0.0008 bpb
- **bigram=12288: expected -0.0005 bpb** (diminishing returns but still significant)

### 3. TrigramHash(4096, dim=64) — Novel
Hash consecutive token triplets into a 4096-bucket embedding table (dim=64), projected to model_dim. This captures 3-gram context that bigrams miss, providing complementary signal. Uses a different hash function (48271/36313/27191 XOR) to minimize correlation with the bigram hash.

### 4. Aggressive SWA (start_frac=0.35, every=40)
More aggressive weight averaging schedule:
- Start collecting earlier (35% vs SOTA 40%) to capture more of the convergence trajectory
- Average more frequently (every 40 steps vs SOTA 50) for smoother weight distributions
- More checkpoints averaged → better quantization-friendly weight distributions

### 5. 5% Magnitude Pruning
Increased from SOTA's 3% to 5%. More aggressive pruning zeros out more small weights, improving zstd compression ratio. The quantization noise from small weights being nonzero is eliminated.

## Hyperparameters

| Parameter | Value | SOTA Value | Delta |
|-----------|-------|------------|-------|
| num_layers | 10 | 10 | same |
| model_dim | 512 | 512 | same |
| mlp_mult | 3.0 | 3.0 | same |
| bigram_vocab | 12288 | 10240 | +2048 |
| trigram_vocab | 4096 | 0 | +4096 (novel) |
| trigram_dim | 64 | — | novel |
| swa_start_frac | 0.35 | 0.40 | -0.05 |
| swa_every | 40 | 50 | -10 |
| prune_frac | 0.05 | 0.03 | +0.02 |
| MLP fc quant | int4 | int5 | novel mixed |
| MLP proj quant | int5 | int5 | same |
| Attn quant | int6 | int6 | same |

## Expected Improvements (Additive)

| Technique | Expected BPB Improvement |
|-----------|------------------------|
| BigramHash 10240 → 12288 | -0.0005 |
| TrigramHash (4096, dim=64) | -0.002 to -0.004 |
| Better SWA schedule | -0.0003 |
| Int4 MLP fc + size savings | -0.001 (more params possible) |
| Better pruning | -0.0002 |
| **Total expected** | **-0.004 to -0.006** |

## Artifact Size Budget

| Component | Estimated Bytes |
|-----------|---------------|
| MLP fc weights (10L, int4+zstd) | ~5.2MB |
| MLP proj weights (10L, int5+zstd) | ~4.2MB |
| Attention weights (10L, int6+zstd) | ~3.8MB |
| Embeddings (FP16) | ~1.0MB |
| BigramHash embed (12288×128) | ~0.8MB |
| TrigramHash embed (4096×64) | ~0.13MB |
| Control tensors + scales | ~0.3MB |
| Code | ~0.05MB |
| **Total** | **~15.5MB** |

## Local Development (MLX)

```bash
# Smoke test on Mac
RUN_ID=anai_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=16384 GRAD_ACCUM_STEPS=4 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 python3 records/track_10min_16mb/AnAI/train_gpt_mlx.py
```

Built upon the work of thwu1 (SOTA), raahilshah, and the parameter-golf community.
