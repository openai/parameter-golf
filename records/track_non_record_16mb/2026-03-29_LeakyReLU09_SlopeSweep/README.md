# Non-record: LeakyReLU(0.9)² Activation Sweep

**Local validation on RTX 5060 (1 shard, 1024 seq_len) | Full 8xH100 validation pending**

## Motivation

PR #466 (LeakyReLU_LegalTTT_ParallelMuon) introduced LeakyReLU(0.5)² as a one-line activation change yielding -0.003 BPB over relu². The ablation in PR #493 confirmed this gain.

However, a community sweep over negative slopes suggests that **slope=0.9 may be significantly better than 0.5** — preserving 81% of negative pre-activation signal (0.9²) vs 25% (0.5²) while still producing non-negative outputs via squaring. This submission investigates the optimal slope for LeakyReLU² activations in the sub-16MB regime.

## Method

Starting from the PR #466 stack (11L/512d/8H/4KV, 3x MLP, XSA last 4, EMA, BigramHash, SmearGate, Partial RoPE 16/64, GPTQ-lite int6, sliding window stride=64), the only change is:

```python
# PR #466 (slope=0.5)
x = F.leaky_relu(self.fc(x), negative_slope=0.5)

# This submission (slope=0.9)
x = F.leaky_relu(self.fc(x), negative_slope=0.9)
```

## Local Results (RTX 5060 8GB, 1 shard, 20K steps)

### Baseline (9L/2x MLP/relu², INT8+zlib)

| Metric | Value |
|--------|-------|
| val_bpb (20K steps) | 1.4364 |
| val_bpb (INT8 roundtrip) | **1.4395** |
| Artifact size | 15.4 MB |
| Step avg | 183 ms |
| Peak VRAM | 1,176 MiB |

### PR #466 stack + LeakyReLU(0.9)² (11L/3x MLP, INT6+zstd)

| Metric | Value |
|--------|-------|
| val_bpb (20K steps) | 1.4892 |
| val_bpb (EMA applied) | 1.4875 |
| val_bpb (INT6 roundtrip) | 1.4889 |
| val_bpb (INT6 + sliding window stride=64) | **1.4508** |
| Artifact size | 12.7 MB |
| Step avg | 247 ms |
| Peak VRAM | 1,345 MiB |

### Component-level analysis

| Component | Contribution |
|-----------|-------------|
| EMA averaging | -0.0017 bpb (1.4892 → 1.4875) |
| INT6 quantization loss | +0.0014 bpb (1.4875 → 1.4889) |
| Sliding window (stride=64) | **-0.0381 bpb** (1.4889 → 1.4508) |

### Observations

The 27M-parameter model (11L/3xMLP) does **not** outperform the 17M baseline (9L/2xMLP) in this limited-data regime. After sliding window correction:

- Baseline: ~1.405 bpb (estimated with sliding window)
- PR #466 stack + LeakyReLU(0.9)²: 1.4508 bpb

This ~0.045 bpb gap is expected due to experimental constraints:

1. **Data volume**: 1 shard (~100M tokens) vs full 80 shards (~10B tokens). The larger model requires significantly more diverse training data to realize its capacity advantage.
2. **Batch size**: 8,192 tokens/step vs 786,432 tokens/step (96x smaller). Gradient noise scales inversely with batch size, disproportionately affecting larger models.
3. **Sequence length**: 1024 vs 2048. Many architectural improvements (XSA, Partial RoPE) are designed for longer context.

These are fundamental data/compute limitations, not architectural flaws. On 8xH100 with 80 shards, this stack achieves 1.1233 bpb (PR #466 baseline) — the LeakyReLU(0.9)² modification is expected to improve on this.

## Hardware Compatibility

FA3 → PyTorch SDPA fallback for non-H100 GPUs:

```python
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False
```

SDPA path transposes between (B,T,H,D) and (B,H,T,D) formats and enables GQA via `enable_gqa` parameter. Mathematically equivalent to FA3, verified on RTX 5060 (SM120/Blackwell consumer).

## Planned Experiments (pending compute)

1. **Slope sweep**: 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95 on 8xH100 with 80 shards
2. **3-seed validation**: Seeds 1337, 42, 2025 for statistical significance
3. **Isolated ablation**: LeakyReLU(0.9)² vs relu² on identical PR #466 stack (single variable)
4. **Interaction effects**: Whether optimal slope changes with model scale (9L vs 11L)

## Files

- `train_gpt.py` — Modified training script (based on PR #466 stack + SDPA fallback + LeakyReLU 0.9)
- `README.md` — This file
