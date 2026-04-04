# Adaptive Focal Loss + Residual Vector Quantization + Progressive Depth Warmup

**Three novel techniques + six proven improvements on the SOTA (PR #549) stack**

> **Status**: Smoke-tested on 1xH100 (Modal). Awaiting 8xH100 SXM verification. Seeking compute access or collaboration.

## Novel Contributions (Original Work)

### 1. Adaptive Focal Cross-Entropy Loss

**Problem**: Standard cross-entropy treats all tokens equally, but easy tokens (articles, spaces) dominate gradients while hard tokens (rare words, unusual patterns) contribute most to BPB.

**Solution**: Dynamically reweight the loss using the model's own confidence:

```python
# Per-token confidence → focal weight
p_t = torch.exp(-token_nll.detach())       # model confidence
focal_weight = (1 - p_t) ** gamma          # hard tokens get higher weight
focal_weight = focal_weight / focal_weight.mean()  # normalize to preserve LR scale
loss = (focal_weight * token_nll).mean()   # focus gradients on hard tokens
```

With `gamma=1.0`: a token predicted at 90% confidence gets 10% weight, while a token at 10% confidence gets 90% weight. This concentrates the limited training budget (~5000 steps) on the most informative tokens.

**Why this is novel**: Focal loss (Lin et al., 2017) exists for classification but has never been applied to language model training in this competition. Our adaptive variant normalizes weights to preserve gradient magnitude compatibility with Muon.

**Expected gain**: -0.005 to -0.015 BPB

### 2. Residual Vector Quantization (RVQ)

**Problem**: Single-pass int6 quantization loses ~0.005-0.010 BPB. With zstd-22, our artifact is only 7MB (9MB headroom in the 16MB budget).

**Solution**: Two-pass quantization that exploits the headroom:

```python
# Pass 1: Standard int6 (GPTQ-lite percentile search)
q6, s6 = quantize_int6_per_row(weight)
reconstructed = dequantize(q6, s6)

# Pass 2: int4 on the residual error
residual = weight - reconstructed
q4, s4 = quantize_residual_int4_per_row(residual)

# At eval: reconstruct from both passes
weight_rvq = dequantize(q6, s6) + dequantize(q4, s4)  # ~10-bit effective precision
```

**Size budget**: int6 base (~7MB zstd) + int4 residual (~3-4MB, low entropy → compresses well) = ~10-11MB total, still within 16MB.

**Why this is novel**: RVQ is standard in audio codecs (SoundStream, EnCodec) but has never been applied to LLM weight compression in this competition. It provides near-lossless quantization within the same artifact budget by exploiting the massive compression headroom from zstd-22.

**Expected gain**: -0.003 to -0.008 BPB (closes 50-80% of the quantization gap)

### 3. Progressive Depth Warmup

**Problem**: With only ~5000 training steps, all 11 layers receive gradients from step 1. But deep layers get near-random gradients early on because shallow layers haven't learned meaningful representations yet. This wastes precious training signal.

**Solution**: Train bottom-up in 3 stages:

```
Stage 0 (steps 0-8%):    Train layers 0-3 only (freeze 4-10)
Stage 1 (steps 8%-25%):  Train layers 0-6   (freeze 7-10)
Stage 2 (steps 25%+):    All 11 layers active
```

Implementation: zero gradients for frozen layer banks after backward, before optimizer step. No architecture changes, no torch.compile impact.

```python
# After backward, before optimizer:
for i in frozen_layers:
    base_model.qo_bank.grad[i].zero_()      # Q weights
    base_model.qo_bank.grad[n + i].zero_()   # Out weights
    base_model.kv_bank.grad[i].zero_()        # K weights
    base_model.kv_bank.grad[n + i].zero_()    # V weights
    base_model.mlp_up_bank.grad[i].zero_()    # MLP up
    base_model.mlp_down_bank.grad[i].zero_()  # MLP down
```

**Why this is novel**: Gradual unfreezing is established in transfer learning (ULMFiT, Howard & Ruder 2018) but has never been applied to **training from scratch** in this competition. It's especially effective here because:
- U-Net skip connections mean stable encoders → stable everything
- Muon's Newton-Schulz orthogonalization amplifies gradient noise
- EMA/SWA make early-step quality crucial

**Expected gain**: -0.005 to -0.010 BPB

## Combined Improvement Stack

| # | Technique | Type | Novel? | Expected Gain |
|---|-----------|------|--------|---------------|
| 1 | **Adaptive Focal Loss** | Training objective | **Yes** | -0.005 to -0.015 |
| 2 | **Residual Vector Quantization** | Post-training compression | **Yes** | -0.003 to -0.008 |
| 3 | **Progressive Depth Warmup** | Training schedule | **Yes** | -0.005 to -0.010 |
| 4 | Polynomial softcap (degree 5) | Architecture | Ported from PR #640 | -0.003 to -0.005 |
| 5 | Z-loss regularization | Training | Ported from PR #640 | -0.001 to -0.002 |
| 6 | YaRN positional encoding | Architecture | Ported from PR #640 | -0.001 to -0.002 |
| 7 | zstd-22 compression | Compression | Standard | Enables RVQ |
| 8 | Sliding eval stride=16 | Evaluation | Ported from PR #640 | -0.002 to -0.005 |
| 9 | FA3/FA2/SDPA fallback | Infrastructure | New | Enables wider testing |

**Conservative total: -0.020 to -0.047 BPB → target ~1.072-1.099 BPB**

## Architecture (Inherited from PR #549)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3x expansion, LeakyReLU(0.5)² |
| Attention | Partial RoPE (16/64), XSA last 4, QK gain=1.5 |
| Embeddings | Tied, SmearGate, BigramHash(1536), VE128 @ layers 9,10 |
| U-Net | 5 encoder + 6 decoder, learned skip weights |
| Optimizer | Parallel Muon (NS5) + AdamW, WD=0.04 |
| Schedule | Warmdown=3500, Late QAT@0.15, EMA(0.997) + SWA |
| TTT | Legal score-first, 3ep SGD, all blocks unfrozen |
| **New: Loss** | **Adaptive Focal CE (gamma=1.0) + Z-loss (1e-4)** |
| **New: Softcap** | **Poly5** |
| **New: RoPE** | **YaRN (max_len=2048)** |
| **New: Quantization** | **RVQ: int6 base + int4 residual** |
| **New: Compression** | **zstd-22** |
| **New: Training** | **Progressive Depth Warmup (3 stages)** |
| **New: Eval stride** | **16** |

## Run Command

```bash
# Full 8xH100 run with all novel techniques
SEED=1337 \
FOCAL_ENABLED=1 FOCAL_GAMMA=1.0 \
RVQ_ENABLED=1 \
PROGRESSIVE_DEPTH=1 PROGRESSIVE_DEPTH_SCHEDULE="0.08,0.25" \
SOFTCAP_TYPE=poly Z_LOSS_WEIGHT=1e-4 \
ROPE_TYPE=yarn YARN_MAX_LEN=2048 \
EVAL_STRIDE=16 \
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
BIGRAM_VOCAB_SIZE=1536 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **PR #549** by @abaybektursun — base SOTA stack
- **PR #640** by @CiprianFlorin-Ifrim — poly5 softcap, Z-loss, YaRN, stride-16
- **PR #414** by @signalrush — GPTQ-lite, EMA
- **PR #461** by @Christopher-Lee-McClendon — TTT protocol
- **PR #399** by @abaybektursun — Parallel Muon
- **PR #493** by @parinzee — LeakyReLU²

Novel techniques (Adaptive Focal Loss, RVQ, Progressive Depth Warmup) by Monisha Kollipara.
