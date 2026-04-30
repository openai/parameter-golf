# AWQ + Cyclic Momentum + ReLU² + 11L Shared — val_bpb ≈ 1.1507

**Track:** 10min / 16MB
**Hardware:** 8×H100 SXM, 600s wallclock
**Model size:** ~15.4 MB (int5 MLP / int6 attn + zstd)
**val_bpb:** 1.1507 ± 0.0016 (3-seed mean, seeds 42, 43, 44)

## Key Innovations

Starting from the community SOTA baseline (thwu1), we introduce four techniques:

| Technique | Description | Impact |
|-----------|-------------|--------|
| **AWQ (Activation-Aware Weight Quantization)** | Scale weight columns by activation importance (alpha=0.5) before quantization. Folds compensation into preceding LayerNorm. Reduces quantization error on high-activation channels. | Quant gap 0.027 → 0.010 bpb |
| **Cyclic Muon Momentum** | Triangle wave between 0.85–0.95 with period=50 steps, replacing fixed 0.99 after warmup. Prevents optimizer from settling into sharp minima. | −0.0045 bpb on 1×H100 |

## Results (8×H100)

| Seed | Steps | Raw val_bpb | Quantized val_bpb | Model Size | Total Size |
|------|-------|-------------|-------------------|------------|------------|
| 42 | 5999 | 1.1605 | 1.1502 | 15.39 MB | 15.45 MB |
| 43 | 6005 | 1.1598 | 1.1494 | 15.46 MB | 15.52 MB |
| 44 | 6000 | 1.1619 | 1.1526 | 15.37 MB | 15.43 MB |
| **Mean** | **6001** | **1.1607** | **1.1507** | **15.41 MB** | |
| **Std** | | **0.0011** | **0.0016** | | |

## Architecture

| Parameter | Value |
|-----------|-------|
| num_layers | 11 (10 unique, last shared) |
| model_dim | 512 |
| num_heads | 8 |
| num_kv_heads | 4 (GQA) |
| mlp_mult | 3.0 (hidden=1536) |
| mlp_activation | relu_sq |
| vocab_size | 1024 |
| train_seq_len | 2048 |
| tie_embeddings | yes |
| logit_softcap | 30.0 |
| rope_base | 10000 |
| rope_dims | 64 (full) |
| bigram_vocab_size | 10240 |
| bigram_dim | 128 |
| skip_connections | U-Net (5 encoder, 6 decoder) |

## Training

| Parameter | Value |
|-----------|-------|
| train_batch_tokens | 786,432 |
| optimizer (matrices) | Muon, lr=0.025, momentum=cyclic 0.85–0.95 |
| optimizer (embeds/scalars) | AdamW, lr=0.035/0.025 |
| warmup_steps | 20 |
| warmdown_iters | 3500 |
| weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| muon_momentum_warmup | 0.92 → cyclic over 1500 steps |
| SWA | start_frac=0.2, every=50 steps |

## Quantization

| Component | Precision |
|-----------|-----------|
| MLP weights | int5 per-row |
| Attention weights | int6 per-row |
| Bigram embeddings | int6 per-row |
| Token embeddings | int8 per-row |
| Control tensors | fp32 passthrough |
| Compression | zstd |

**AWQ:** Before quantization, run 8 calibration batches through the model. For each Linear layer, compute per-channel activation magnitude `s = act.abs().mean(dim=(0,1))`. Scale weight columns by `s^0.5`, fold inverse into preceding LayerNorm. This protects high-activation channels from quantization error.

## Evaluation

Sliding window evaluation with stride=64, batch_seqs=64.

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

## Development Journey

This submission emerged from 21 experiments on 1×H100 and 1×A40, systematically testing:
- Multi-token prediction (MTP) — marginal gains, size overhead
- Curriculum learning — incompatible with torch.compile
- Test-time training (TTT) — promising with partial RoPE, but eval time too long
- Various quantization strategies (GPTQ-lite, layer-aware, EMA) — AWQ was the clear winner
- Architectural variations (wider, value embeddings, partial RoPE) — diminishing returns

The final recipe: simple architecture + smart optimization (cyclic momentum) + smart quantization (AWQ).
