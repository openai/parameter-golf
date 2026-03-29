# Turbo-Muon + EngramLite + Parameter Banking + GPTQ Mixed-Precision

**val_bpb: TODO** (3-seed mean, std TODO) | **~15.6 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | step_avg | steps | val_bpb (SW) | val_bpb (full) | Artifact bytes |
|------|----------|-------|-------------|----------------|----------------|
| 42   | TODO     | TODO  | TODO        | TODO           | TODO           |
| 1337 | TODO     | TODO  | TODO        | TODO           | TODO           |
| 2025 | TODO     | TODO  | TODO        | TODO           | TODO           |
| **Mean** | **TODO** | **TODO** | **TODO** | **TODO** | |

## Summary

An 11-layer GPT language model combining six key innovations over the PR #609 baseline, targeting the 16MB artifact budget with zero selective pruning at MLP 3.5x width. Development-run benchmark: **1.1119 val_bpb (sliding window)** on 1xH100.

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

### Mimetic V-O Initialization

Output projections initialized as `O_h = -alpha * V_h` per head (alpha=0.05), creating a small residual-like identity at init for improved early training stability.

### GPTQ Mixed-Precision Quantization

Post-training compression pipeline:
1. Hessian collection with calibration batches (within training budget via `gptq_reserve_ms=14000`)
2. Dynamic mixed-precision bit allocation: Hessian trace sensitivity ranks tensor groups, greedy promotion to int7/int6
3. GPTQ with Cholesky error compensation and column permutation by descending Hessian diagonal
4. Late QAT (soft-round) activated when LR scale < 15%, with alpha ramping 1 to 16
5. Selective pruning of low-magnitude quantized values (binary search to fit 16MB)
6. Brotli + byte-shuffle compression (byte-shuffle groups same-position bytes for better entropy coding)

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3.5x with LeakyReLU(0.3)^2 |
| EngramLite | 2 heads x 2 orders, 8192 buckets |
| Skip connections | U-Net sigmoid-gated |
| RoPE | Partial (16 dims) |
| ValueEmbedding | Layers 9-10 |
| SmearGate | Causal shift blending |
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
- **SmearGate + BigramHash**: [PR #198](https://github.com/openai/parameter-golf/pull/198) and related submissions
- **Polar Express coefficients**: Amsel et al. (arXiv:2505.16932)
- **GPTQ approach**: [PR #634](https://github.com/openai/parameter-golf/pull/634) Hessian-aware quantization

## TODO Before Submission

- [ ] Run 3 seeds on 8xH100 (42, 1337, 2025) and collect logs
- [ ] Fill in results table with actual val_bpb, step times, artifact sizes
- [ ] Update submission.json with actual val_bpb and bytes_total
- [ ] Verify 3-seed mean beats SOTA by >= 0.005 nats
- [ ] Attach log files as `train_seed42.log`, `train_seed1337.log`, `train_seed2025.log`
