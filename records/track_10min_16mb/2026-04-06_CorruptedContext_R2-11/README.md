# Record: Corrupted Context + EMA + QAT + Per-Group Int6 Bit-Packed Quantization

**val_bpb: TBD** (3-seed mean) | **~10.3 MB** (est) | 8xH100 SXM, 600s | No TTT

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **int6 packed BPB** | Artifact |
|------|-------|---------|---------------|---------------------|----------|
| 42 | TBD | TBD | TBD | **TBD** | TBD |
| 137 | TBD | TBD | TBD | **TBD** | TBD |
| 256 | TBD | TBD | TBD | **TBD** | TBD |
| **Mean** | | | | **TBD** | |

---

## Key Innovations

### 1. Corrupted Context Training (Novel)

During training, 10% of input tokens are replaced with uniformly random tokens (first position never corrupted). This bridges the train/inference exposure bias gap: the model learns to make predictions even when some context tokens are noisy, improving robustness and generalization.

This technique is orthogonal to standard dropout (which operates on feature dimensions). Corrupted context operates on the **sequence dimension**, forcing the model to build redundant causal paths rather than over-relying on any single context position.

**Screening result**: On 1-shard screening, corrupted context training (val_bpb=1.3004) beat both reproduced SOTA submissions (1.3315, 1.3471) by a significant margin.

### 2. Per-Group Int6 Bit-Packed Quantization (Novel)

Standard int6 quantization uses one scale per row (up to 1536 weights), where a single outlier ruins resolution for the entire row. We use **per-group-64 scaling**: each group of 64 weights gets its own scale, adapting to the local weight range.

Values are **6-bit packed** (4 values per 3 bytes) instead of stored in 8-bit containers, reducing raw size by 25%. Combined with lzma preset=9 compression, this achieves **~10MB for 32.7M params** (0.31 bytes/param compressed).

Best-of-5 percentile search per group selects the optimal clipping point for each group independently, minimizing reconstruction error.

### 3. Late QAT (Quantization-Aware Training)

When the learning rate scale drops below 0.15 during warmdown, we enable **Straight-Through Estimator (STE)** fake quantization in every CastedLinear forward pass. This simulates per-group-64 int6 rounding during training, allowing the model to adapt its weights to be quantization-friendly before the actual post-training quantization.

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | **4x (2048)** with LeakyReLU(0.5)^2 | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Attention | **XSA on all 11 layers** | [#198](https://github.com/openai/parameter-golf/pull/198), [#478](https://github.com/openai/parameter-golf/pull/478) |
| BigramHash | 3072 x dim=112 | [#162](https://github.com/openai/parameter-golf/pull/162), [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| RoPE | **Partial (16/64 dims)** | [#315](https://github.com/openai/parameter-golf/pull/315) |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Value residual | v + sigmoid(alpha) * v0 | [#549](https://github.com/openai/parameter-golf/pull/549) |
| Weight avg | **EMA(0.997)** | [#401](https://github.com/openai/parameter-golf/pull/401) |
| Quantization | **Per-group-64 int6 bit-packed + lzma** | **This work** |
| QAT | **Per-group-64 STE at LR scale < 0.15** | [#286](https://github.com/openai/parameter-golf/pull/286) |
| Warmdown | **3500 iterations** | [#364](https://github.com/openai/parameter-golf/pull/364) |
| Optimizer | Muon (NS5) + Adam, **WD=0.04, momentum=0.99** | [#399](https://github.com/openai/parameter-golf/pull/399) |
| Grad clip | **0.3** | Standard |
| Corrupted context | **10% random token replacement** | **This work** |

**Total parameters**: ~32.7M (enabled by int6 compression headroom — same 16MB budget holds 21% more params)

## Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All features are enabled by default. No env vars needed.

## Lineage

```
Baseline (9L/2x, relu^2, 1.2244 BPB)
    +-- LeakyReLU(0.5)^2 + 11L/4x MLP (#493)
    +-- BigramHash 3072x112 (#162, #1019)
    +-- XSA all 11 layers (#198, #478)
    +-- Value residual (#549)
    +-- U-Net skips (#289)
    +-- EMA(0.997) (#401)
    +-- Partial RoPE 16/64 (#315)
    +-- Warmdown 3500 (#364)
    +-- Muon WD=0.04, momentum=0.99 (#399)
    +-- Corrupted context 10% (this work)
    +-- Per-group-64 int6 bit-packed + QAT (this work)
```

## Requirements

```bash
pip install torch>=2.4.0 numpy sentencepiece
```
No Flash Attention 3 required — uses standard PyTorch SDPA.
