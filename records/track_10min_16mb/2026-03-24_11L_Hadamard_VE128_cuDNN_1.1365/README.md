# 11L + Hadamard Rotation + VE128 + cuDNN SDPA (val_bpb: 1.1364)

## Key Innovation: Hadamard Rotation for Int6 Quantization

Walsh-Hadamard rotation applied to weight matrices before int6 per-row quantization. The orthogonal rotation spreads outlier values uniformly across all dimensions, improving zstd compressibility from 1.70x to 1.77x while reducing the quantization gap from 0.0093 to 0.0084 BPB. This is the first application of rotation-based quantization (QuIP-family) in this competition. No other open or merged PR uses this technique.

The 0.07x compression improvement translates to 530KB of recovered headroom within the 16MB artifact budget, directly enabling the addition of Shared Value Embeddings (VE128) which previously overflowed at 44KB headroom.

Negative result: Full GPTQ (Hessian-calibrated quantization) provides zero additional benefit when combined with Hadamard rotation. The rotation already makes weight distributions sufficiently uniform for simple abs-max quantization. This confirms that Hadamard rotation and GPTQ are substitutes, not complements, at int6 precision.

## Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared activation
- Exclusive Self-Attention (XSA) on last 4 layers (GQA-aware)
- Partial RoPE (16/64 dims)
- Layer-norm scale factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate + BigramHash (2048 buckets, inner_dim=128)
- Shared Value Embedding (dim=128, layers 9 and 10, per-layer learned scales)
- cuDNN SDPA backend (FlashAttention 3 conditional fallback)
- Logit softcap 30.0, tied embeddings
- Orthogonal init with projection scaling by 1/sqrt(2*num_layers)

## Training

- Muon optimizer (matrix params): lr=0.025, momentum=0.99 (warmup 0.92 to 0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 524,288 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wall-clock based, cosine schedule)
- EMA decay=0.997 (continuous, every step)

## Quantization

- Int6 per-row with Hadamard rotation (block-diagonal Walsh-Hadamard on column dimension)
- Abs-max scaling (no percentile clipping, no GPTQ)
- Control tensors (scales, gates, VE scales) in FP16
- zstd level 22 compression

## Ablation

| Config | Sliding BPB | Compression | Headroom | Quant Gap |
|--------|-------------|-------------|----------|-----------|
| Baseline (S5-7, no Hadamard, no VE) | 1.1372 | 1.70x | 44KB | 0.0093 |
| + Hadamard rotation | 1.1377 | 1.78x | 712KB | 0.0091 |
| + VE128 (enabled by headroom) | **1.1364** | 1.77x | 530KB | 0.0084 |
| + GPTQ on top of Hadamard | 1.1401+ | 1.73x | 201KB | 0.0088 |

Hadamard rotation enables VE128 by freeing 668KB of artifact headroom. GPTQ adds no value when Hadamard is present.

## Methodology: CPU Parameter Probe

Hyperparameter selection was guided by a CPU-based parameter sweep engine (Go, 80-core Modal) that estimates sliding BPB from architecture configuration without GPU training. The probe uses three independent estimation approaches (K-Nearest Neighbor, bounded parametric, relative delta) calibrated against 15 GPU training runs across sessions 2-7.

This is a directional methodology, not a high-precision predictor. The probe narrows the search space by eliminating configurations that are unlikely to fit artifact constraints or improve BPB, reducing the number of expensive GPU runs needed. In our workflow, it guided the path from 9L to 11L MLP1536, correctly flagged FA3 compression overflow and 12L artifact limits, and identified the embed_lr=0.035 configuration that produced our best result -- all confirmed by subsequent GPU training.

The approach is exploratory and specific to our calibration data. It demonstrates how lightweight CPU-based pre-screening can complement GPU experimentation when compute is constrained, and could be adapted to other search spaces with different calibration sets.

## Additional Findings

- **Late QAT was dead code in all prior work**: The `CastedLinear.qat_enabled` class attribute change was shadowed by instance attributes set during init. QAT-STE never activated during training in any session. Removing the dead QAT guard from CastedLinear.forward() eliminated a torch.compile guard, reducing step time from 70.6ms to 65.7ms (7% throughput gain).
- **FA3 compression penalty**: FlashAttention 3 produces weight distributions that compress 1.8% worse with zstd-22 (1.67x vs 1.70x cuDNN). The throughput gain (65ms vs 70.6ms) does not compensate at MLP1536 due to artifact overflow.
- **FP16 control params**: Storing scale/gate tensors as FP16 instead of FP32 is lossless for eval (values are cast to bfloat16 in forward). Saves 50KB raw payload.
- **INT6 bigram projection**: Quantizing the 128x512 BigramHash projection to int6 (vs FP16 passthrough) improves zstd compression by 0.09x. The quantization noise is negligible for this small embedding projection.

## Results (3 seeds)

| Seed | Steps | Pre-quant | Sliding BPB | Artifact Bytes | Compression |
|------|-------|-----------|-------------|----------------|-------------|
| 1337 | 8098 | 1.1512 | 1.1364 | 15,618,718 | 1.75x |
| 42 | 8102 | 1.1513 | 1.1361 | 15,629,540 | 1.75x |
| 2024 | 7960 | 1.1521 | 1.1370 | 15,600,361 | 1.76x |

**Mean: 1.1365 +/- 0.0005 BPB.** All artifacts under 16MB. 27,038,810 parameters.

## Run

```bash
NUM_LAYERS=11 MLP_MULT=3 XSA_LAYERS=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_DIM=128 VE_LAYERS=9,10 EMA_DECAY=0.997 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
WARMDOWN_ITERS=3500 USE_CUDNN_SDPA=1 \
torchrun --nproc_per_node=8 train_gpt.py
```

Erick Aleman | EA Cognitive | www.eacognitive.com | github.com/eacognitive
