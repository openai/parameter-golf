# 11L Partial RoPE + XSA4 + VE128 + Tight SWA + Late QAT + GPTQ-lite

## Score: val_bpb = 1.1804 (post-quant, single seed)

Trained on 8×H100 SXM in 615 seconds. 15.95MB artifact (int6+zstd-22).

## Approach

Combines the PR #374 SOTA stack with MLP width reduction (1408 vs 1536) to fit under 16MB, plus GPTQ-lite quantization optimization.

### Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- MLP hidden=1408 (2.75× expansion), relu-squared activation
- **Partial RoPE** (16/64 dims): Only 25% of head dims get rotary embeddings. The remaining 75% are position-free, improving generalization.
- **LN Scale** (1/sqrt(layer_idx+1)): Damps RMSNorm output in deeper layers, stabilizing gradient flow.
- **XSA** on last 4 layers: Exclusive Self Attention removes self-value bias via GQA-aware orthogonal projection. Zero new parameters, ~2ms/step.
- **Shared Value Embedding** (dim=128, layers 9,10): Single embedding table projected to KV dim, added to V in selected layers. Per-layer learned scales.
- SmearGate: Learned per-dim gate blending current + previous token embeddings.
- U-Net skip connections (5 encoder, 6 decoder), tied embeddings, logit softcap 30.

### Training
- Muon optimizer: lr=0.025, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- AdamW: embed_lr=0.035, scalar_lr=0.025, WD=0.04
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3000 iters (wallclock-based), grad_clip=0.3
- **Tight SWA**: Uniform average of checkpoints collected every 50 steps when lr_scale < 0.2 (6 checkpoints total). Zero quality penalty vs non-SWA.
- **Late QAT**: STE int6 fake-quantization activated when lr_scale < 0.1 (step 4070). LR halved at activation to avoid disrupting converged weights.

### Quantization
- **GPTQ-lite**: Per-tensor clip ratio search (5 candidates: 0.9999, 0.99999, 0.999999, 0.9999984, 1.0). Selects the clip percentile that minimizes reconstruction error L2. Zero training cost.
- Int6 step=4 rounding on layers 1-9 (64 distinct values for better compression)
- Int8 on layers 0 and 10 (input/output quality)
- FP16 tied embeddings (never quantized)
- zstd level 22 compression

## Key Metrics

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1770 |
| **Post-quant val_bpb** | **1.1804** |
| Quant gap | +0.0034 |
| Steps completed | 4,071 |
| Step time | 137ms avg (151ms after Late QAT) |
| Model parameters | 25,224,291 |
| Artifact size | 15,949,473 bytes (15.95 MB) |
| Peak GPU memory | 20,590 MiB |

## Convergence

| Step | val_bpb | train_time |
|------|---------|-----------|
| 1000 | 1.3246 | 136s |
| 2000 | 1.2551 | 274s |
| 3000 | 1.2139 | 413s |
| 4000 | 1.1793 | 551s |
| 4071 | 1.1770 | 615s (cap) |

## Lessons Learned

1. **MLP hidden=1408 > 1536 for artifact-constrained models**: Narrower MLP fits in 16MB with int6+zstd while enabling ~33% more training steps (137ms vs 178ms/step). The extra steps more than compensate for reduced per-step capacity.

2. **Late QAT timing matters**: Activating at lr_scale<0.1 (last ~1% of training) gives only 1 step of QAT adaptation. Earlier activation (lr_scale<0.2) would give more adaptation time but risks disrupting Muon momentum.

3. **Tight SWA (scale<0.2) eliminates SWA quality penalty**: Standard SWA (scale<0.5) averages stale early-warmdown checkpoints that hurt final quality. Restricting to scale<0.2 produces weight averaging with zero quality loss.

4. **GPTQ-lite clip search is free**: Trying 5 clip ratios per tensor during quantization costs ~2s total and reduces reconstruction error without any training cost.

## Command

```bash
pip install --break-system-packages zstandard

RUN_ID=pr374_8x_v2 MLP_HIDDEN=1408 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Status

Single seed (1337). Non-record submission — val_bpb 1.1804 does not beat SOTA 1.1428 by the required 0.005 margin. Submitted as a non-record contribution documenting the systematic combination of frontier techniques.
