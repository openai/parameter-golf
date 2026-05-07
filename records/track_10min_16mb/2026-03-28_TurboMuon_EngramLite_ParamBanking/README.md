# Turbo-Muon + EngramLite + Parameter Banking + GPTQ Mixed-Precision

**val_bpb: 1.1091** (3-seed mean, std 0.0005) | **~15.3 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | val_bpb (SW) | val_bpb (full) | Artifact bytes |
|------|----------|-------|-------------|----------------|----------------|
| 42   | 93.26ms  | 6284  | 1.1086      | 1.1324         | 15,992,528     |
| 1337 | 93.11ms  | 6295  | 1.1090      | 1.1328         | 15,993,413     |
| 2025 | 93.11ms  | 6294  | 1.1096      | 1.1335         | 15,993,904     |
| **Mean** | **93.16ms** | **6291** | **1.1091** | **1.1329** | |

## Summary

An 11-layer GPT language model combining seven key innovations over the PR #609 baseline, targeting the 16MB artifact budget at MLP 3.5x width. Development-run benchmark: **1.1119 val_bpb (sliding window)** on 1xH100.

## Key Innovations

### Turbo-Muon Optimizer

A variant of the Muon optimizer with three enhancements that reduce Newton-Schulz iterations from 5 to 4:

- **AOL Preconditioning** -- Gershgorin-based diagonal scaling contracts the singular value range before Newton-Schulz iteration, allowing the first NS step to be skipped.
- **Polar Express Coefficients** -- Optimal degree-5 polynomial coefficients from Amsel et al. (arXiv:2505.16932), applied per-iteration rather than fixed.
- **Post-NS row_col Normalization** -- After orthogonalization, rows then columns are normalized. This consistently outperforms row-only or no normalization.

### EngramLite Hash Embeddings

Multi-head prime-based hash embedding that captures bigram and trigram statistics without explicit tokenizer changes:
- 2 heads x 2 orders (bigram + trigram) with 8192 hash buckets
- Projects to model_dim through a learned sigmoid gate
- Adds character-level context at minimal parameter cost

### Parameter Banking

All per-layer linear weights stored in contiguous 3D tensors (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`). This enables batched Newton-Schulz orthogonalization via `torch.bmm`, dramatically reducing Muon optimizer overhead compared to per-layer iteration.

### U-Net Skip Connections

Encoder/decoder structure with learned sigmoid-gated skip connections. Gates start at `sigmoid(0) = 0.5` and learn per-dimension blending, preventing gradient shortcutting at initialization.

### ValueEmbedding

Reinjects token identity into attention values at deep layers (9, 10). Projects vocabulary embeddings to kv_dim with per-layer learned scaling, helping the model maintain token-level information through deep attention stacks.

### SmearGate

`F.pad`-based causal shift blending each token with its predecessor, providing free unigram context at zero attention cost.

### XSA (Cross-Sequence Attention) — All Layers

Efficient XSA applied to all 11 layers (`XSA_LAST_N=11`). Subtracts the self-value projection from attention output via GQA-aware reshape (no `repeat_interleave`), encouraging the model to attend to context rather than the current token's own representation.

### Mimetic V-O Initialization

Output projections initialized as `O_h = -alpha * V_h` per head (alpha=0.05), creating a small residual-like identity at init for improved early training stability.

### Additional Architecture Details

- **Partial RoPE** — Rotary position embeddings applied to only 16 of 64 head dimensions (`ROPE_DIMS=16`). Remaining dimensions are position-free, giving the model both positional and position-invariant feature channels.
- **LN Scale** — Layer norm outputs scaled by `1/sqrt(layer_idx + 1)`, stabilizing deeper layers by reducing activation magnitudes proportional to depth.
- **Logit Softcap** — `softcap * tanh(logits / softcap)` with softcap=30.0 prevents extreme logit values during training.
- **GQA** — Grouped Query Attention with 8 query heads and 4 KV heads (2:1 grouping), reducing KV cache and parameter count.
- **Tied Embeddings** — Input and output embeddings share weights, saving parameters.
- **QK Gain** — Per-head learnable query scaling initialized to 1.5, allowing the model to tune attention sharpness per head.

### GPTQ Mixed-Precision Quantization

Compression pipeline with Hessian collection performed within the 600s training budget (`gptq_reserve_ms=9000` deducted from training wallclock before training begins):

1. **Hessian collection** — 64 calibration batches run through a non-banked model copy to collect per-layer `H = X^T X` approximations, all-reduced across ranks. This runs within the reserved 14s carved from the training budget.
2. **Dynamic mixed-precision bit allocation** — Base quantization is **int5** for all weight groups. Hessian trace sensitivity ranks tensor groups (by layer × attn/mlp), then a greedy allocator selectively **promotes the most sensitive groups to int6 or int7** until the estimated compressed artifact size approaches the 16MB target minus 2% pruning headroom.
3. **GPTQ quantization** — Hessian-aware Cholesky error compensation for 2D weight matrices. Columns permuted by descending Hessian diagonal for optimal error propagation. Falls back to percentile search on Cholesky failure.
4. **Late QAT (soft-round)** — Quantization-aware training activated when LR scale drops below 15%, with soft-round sigmoid alpha ramping 1→16 over the QAT phase. Provides real gradient signal through quantization grid points.
5. **Selective pruning** — Post-GPTQ, values with `|q| ≤ 2` ranked by reconstruction error impact. Binary search with fast (zlib-1) / real (brotli-11) calibration finds the minimal prune count to fit 16MB.
6. **Brotli + byte-shuffle compression** — Byte-shuffle preprocessing reorders tensor bytes by significance position before brotli compression (quality=11) for optimal entropy coding.

### Code Shrinking

The submission `train_gpt.py` is a compressed self-extracting wrapper generated by
`Shrink/shrink.py`. The pipeline: AST dead-code removal → pyminify (strip comments,
whitespace, type hints, rename identifiers) → LZMA + base85 self-extracting `exec()`.

- **Human-readable source**: `train_gpt_human.py` (123 KB)
- **Shrunk submission**: `train_gpt.py` (24 KB)
- **Code budget freed**: ~99 KB → more artifact space for model weights → less pruning

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3.5x with LeakyReLU(0.3)^2 |
| XSA | All 11 layers |
| EngramLite | 2 heads x 2 orders, 8192 buckets |
| Skip connections | U-Net sigmoid-gated |
| RoPE | Partial (16 of 64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| Logit Softcap | 30.0 |
| ValueEmbedding | Layers 9-10 |
| SmearGate | Causal shift blending |
| Embeddings | Tied input/output |
| Vocab | 1024 BPE, seq 2048 |

### Optimizer

| Param group | LR | Notes |
|---|---|---|
| Bank weights (Muon) | 0.025 | momentum=0.99, WD=0.04 |
| Embeddings (Adam) | 0.6 | betas=(0.7, 0.95), WD=0.04 |
| Head/tied embed (Adam) | 0.035 | betas=(0.7, 0.95) |
| Scalars (Adam) | 0.025 | betas=(0.9, 0.95) |

### Weight Averaging

- **SWA** -- Float32 accumulation every 50 steps after 20% of training
- **EMA** -- Decay=0.997, lerp_ single-kernel updates

## Dependencies

Requires `brotli>=1.1` (included in `requirements.txt`). The code gracefully falls back to lzma > zlib if brotli is missing, but brotli is needed to maximize model capacity within the 16MB budget.

## Run Command

```bash
# 8xH100 (competition)
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Single GPU (development)
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

All hyperparameters are configured via environment variables. Defaults match the competition-optimized configuration. Key overrides:

```bash
# Override examples
NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3.5 \
MUON_MOMENTUM=0.99 MUON_WD=0.04 MUON_POST_NORM=row_col \
SWA_ENABLED=1 EMA_ENABLED=1 MIXED_PRECISION=1 LATE_QAT=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Snapshot Workflow

Train once, iterate on compression without retraining:

```bash
# Step 1: Train and save snapshot
SNAPSHOT_POST_HESSIAN=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Step 2: Load snapshot, re-run compression + eval only
LOAD_SNAPSHOT=snapshot_post_hessian.pt torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Credits

- **Base recipe**: [PR #609](https://github.com/openai/parameter-golf/pull/609) (1.1154 bpb baseline)
- **Muon optimizer**: Inspired by [PR #399](https://github.com/openai/parameter-golf/pull/399) parameter banking approach
- **LeakyReLU^2**: [PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #518](https://github.com/openai/parameter-golf/pull/518)
- **XSA**: [PR #265](https://github.com/openai/parameter-golf/pull/265), [PR #287](https://github.com/openai/parameter-golf/pull/287)
- **SmearGate + BigramHash**: [PR #198](https://github.com/openai/parameter-golf/pull/198) and related submissions
- **Polar Express coefficients**: Amsel et al. (arXiv:2505.16932)
- **GPTQ approach**: [PR #634](https://github.com/openai/parameter-golf/pull/634) Hessian-aware quantization

