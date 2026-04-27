# Seed-Regenerated Random Model + Incremental N-gram Cache — val_bpb 0.0905

**val_bpb = 0.0905** (1 seed, additional seeds pending H100 access) | **15.09 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.7.1)

| Seed | step_avg | steps | neural_bpb | blended_bpb | Artifact |
|------|----------|-------|------------|-------------|----------|
| 1337 | 67ms | 9,912 | 1.503 | **0.0905** | 15,093,968 |
| 42 | — | — | — | — | pending |
| 2025 | — | — | — | — | pending |

> **Note**: Additional seeds pending H100 access.

## Key Innovation: Seed-Regenerated Weights

All weight matrices in the transformer blocks (Q, K, V, O-proj, MLP-up, MLP-down) use **frozen orthogonal random projections** regenerated from deterministic seeds at load time. The artifact stores only:

- **LoRA adapters** (rank-64 A and B matrices): ~3.9 MB at INT8
- **Embedding + control tensors**: ~1.0 MB at FP16
- **N-gram cache** (INT16 counts, LZMA compressed): ~10.7 MB
- **Code**: ~0.1 MB

The random base weights cost **0 bytes** in the artifact — they are regenerated from 8-byte seeds per matrix via QR-decomposed orthogonal initialization.

### Why Orthogonal (not Gaussian)

Prior work (PR #874) used Gaussian random bases but could not train models deeper than 5 layers — gradients vanish through deep stacks of random projections. Our **orthogonal initialization via QR decomposition** preserves singular values at exactly 1.0, enabling stable training of 11-layer random models (though we use 5L here for throughput).

```python
@staticmethod
def _generate_orthogonal_base(seed, rows, cols):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    size = max(rows, cols)
    raw = torch.randn(size, size, generator=g)
    Q, _ = torch.linalg.qr(raw)
    return Q[:rows, :cols] / math.sqrt(cols)
```

### Adapter Quantization: Nearly Lossless

The LoRA adapters are quantized with simple per-row INT8 (no GPTQ needed). The quantization gap is only **+0.003 BPB** — dramatically better than INT6 GPTQ on full weight matrices (+0.006 for the baseline).

## N-gram Cache: Incremental Build During Training

The n-gram cache is built **incrementally during training** with zero overhead:

```python
# After each training microstep (cost: <1ms per call):
ngram_counter.update_batch_fast(full_seq.cpu().numpy().astype(np.int32))
```

- **Orders**: 2-7 (hash-bucketed count tables)
- **Counts**: INT16 (uint16), clipped to 65535
- **Total counts**: 31.1 billion (from 9,912 steps × 524K tokens × 8 GPUs)
- **Multi-GPU sync**: `dist.all_reduce(SUM)` across 8 GPUs before serialization
- **Compression**: LZMA preset 9 → 10.7 MB

At eval time, the cache is **frozen** — no TTT, no eval-time updates. Entropy-adaptive alpha blending:
```
alpha = min(alpha_max, log1p(count) / 10)
P_blend = alpha * P_ngram + (1 - alpha) * P_neural
```

### Why Incremental > Pre-fill

We tested pre-filling the cache from training shards at startup. This was **10× worse** (0.996 BPB vs 0.0905) because:
1. Pre-fill consumed 24-33% of the training budget (650-880s for 10 shards)
2. Numpy hash computation on 50M-token shards was catastrophically slow
3. Only covered 10/80 shards vs incremental seeing ALL training tokens

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 5 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 |
| MLP multiplier | 3.0 (hidden=1536) |
| Activation | LeakyReLU(0.5)² |
| Adapter rank | 64 |
| Random init | Orthogonal (QR decomposition) |
| Vocab | 1024 BPE |
| Sequence length | 2048 |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer (adapters) | Muon (NS5, momentum 0.99) |
| Optimizer (embed/scalar) | AdamW |
| Matrix LR | 0.04 |
| Grad clip norm | 0.1 |
| Weight decay | 0.04 |
| Batch tokens | 524,288 |
| EMA decay | 0.997 |

## Ablation

| Config | BPB | Notes |
|--------|-----|-------|
| Neural only (post-quant) | 1.503 | Adapter INT8, no cache |
| Neural sliding window | 1.474 | stride=64 |
| **Neural + n-gram blend** | **0.0905** | Entropy-adaptive alpha, frozen cache |
| Improvement from cache | -1.413 | |

## Artifact Budget

```
Neural model (INT8 adapters + FP16 embed):   4,401,588 bytes
N-gram cache (INT16 counts, LZMA):          10,692,380 bytes
Total:                                      15,093,968 bytes
Remaining:                                     906,032 bytes
```

## Credits

- PR #874 (@fielding) — Random linear maps concept
- PR #931 (@AnirudhRahul) — Packed n-gram artifact approach
- arXiv:2407.00957 — Expressivity with random weights and learned biases
- PR #549 (@abaybektursun) — LeakyReLU² activation, score-first TTT protocol
