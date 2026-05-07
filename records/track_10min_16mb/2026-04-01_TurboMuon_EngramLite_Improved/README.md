# Turbo-Muon + EngramLite(10240) + VE(8,9,10) + Higher LR

**val_bpb: 1.1431** (3-seed mean, std 0.0007) | **~15.99 MB** | 8xH100 SXM, 600s

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | val_bpb (sliding) | val_bpb (roundtrip) | Artifact bytes |
|------|----------|-------|--------------------|---------------------|----------------|
| 1337 | 106.74ms | 5,538 | **1.1425** | 1.1657 | 15,988,293 |
| 42   | 106.09ms | 5,572 | **1.1438** | 1.1669 | 15,978,184 |
| 2024 | 106.00ms | 5,576 | **1.1431** | 1.1652 | 15,985,158 |
| **Mean** | **106.28ms** | **5,562** | **1.1431** | **1.1659** | |

## Summary

An 11-layer GPT language model based on the [PR #1089](https://github.com/openai/parameter-golf/pull/1089) Turbo-Muon + EngramLite stack, with hyperparameter tuning focused on convergence speed and n-gram coverage within the 16MB artifact budget.

## Changes from PR #1089

| Parameter | PR #1089 | This | Rationale |
|-----------|----------|------|-----------|
| `MATRIX_LR` | 0.025 | **0.030** | Faster convergence within 600s budget |
| `SCALAR_LR` | 0.025 | **0.030** | Matched to matrix LR |
| `WARMDOWN_ITERS` | 3500 | **4500** | Longer warmdown for smoother weight averaging |
| `MUON_MOMENTUM_WARMUP_STEPS` | 1500 | **1000** | Reach target momentum 0.99 faster |
| `VE_LAYERS` | 9,10 | **8,9,10** | Additional token identity injection at layer 8 |
| `NGRAM_BUCKETS` | 8192 | **10240** | Wider n-gram hash coverage |
| `NGRAM_DIM_PER_HEAD` | 32 | **48** | Richer n-gram embeddings |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3.5x with LeakyReLU(ASQU v3 per-layer slopes)^2 |
| XSA | All 11 layers |
| EngramLite | 2 heads x 2 orders (bigram+trigram), **10240 buckets**, dim **48**/head |
| Skip connections | U-Net sigmoid-gated |
| RoPE | Partial (16 of 64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| Logit Softcap | 30.0 |
| ValueEmbedding | **Layers 8, 9, 10** |
| SmearGate | Causal shift blending |
| Embeddings | Tied input/output |
| Vocab | 1024 BPE, seq 2048 |

### Optimizer

| Param group | LR | Notes |
|---|---|---|
| Bank weights (Turbo-Muon) | **0.030** | momentum=0.99, WD=0.04, NS=4, post_norm=row_col |
| Embeddings (Adam) | 0.6 | betas=(0.7, 0.95), WD=0.04 |
| Head/tied embed (Adam) | 0.035 | betas=(0.7, 0.95) |
| Scalars (Adam) | **0.030** | betas=(0.9, 0.95) |

### Quantization

- GPTQ with Hessian-aware Cholesky error compensation (9s reserved from training budget)
- Dynamic mixed-precision: int5 base for all 66 weight groups (0 promoted to int6/int7)
- Selective pruning: 20.5% of +/-1,+/-2 values pruned to fit 16MB
- Brotli-11 + byte-shuffle compression
- Late QAT with soft-round sigmoid alpha ramp (threshold=0.15)

### Weight Averaging

- SWA: float32 accumulation every 50 steps after warmdown threshold (18 checkpoints)
- EMA: decay=0.997

## Key Observations

The increased EngramLite (10240x48) and VE on 3 layers added ~950K parameters (31.6M vs 30.7M in PR #1089), pushing the model to 16.36MB pre-compression. This forced all weight groups into int5 with 0 promotions to int6/int7, and required 20.5% selective pruning. The aggressive quantization + pruning likely offset the gains from wider n-gram coverage and additional VE layer.

**Lesson learned**: Within the 16MB budget, parameter count increases must be carefully balanced against quantization headroom. The sweet spot for EngramLite may be closer to 8192x32 (PR #1089 default) which allows int6 promotions for sensitive layers.

## Run Command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base recipe**: [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @Bortlesboat — Turbo-Muon + EngramLite + Parameter Banking + GPTQ Mixed-Precision
- **LeakyReLU^2**: [PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #518](https://github.com/openai/parameter-golf/pull/518)
- **XSA**: [PR #265](https://github.com/openai/parameter-golf/pull/265), [PR #287](https://github.com/openai/parameter-golf/pull/287)
- **SmearGate + BigramHash**: [PR #198](https://github.com/openai/parameter-golf/pull/198)
- **Polar Express coefficients**: Amsel et al. (arXiv:2505.16932)
- **GPTQ approach**: [PR #634](https://github.com/openai/parameter-golf/pull/634)
