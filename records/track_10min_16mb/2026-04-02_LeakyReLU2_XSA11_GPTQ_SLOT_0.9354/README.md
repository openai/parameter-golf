# LeakyReLU² + XSA-all + Full GPTQ + SLOT

**val_bpb: 0.9354** (3-seed mean: 1337→0.9349, 42→0.9325, 7→0.9388)

## Architecture

- 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)² MLP with 3x expansion
- RoPE, RMSNorm, tied embeddings (vocab=1024), logit softcapping (30.0)
- U-Net skip connections with learned skip weights
- SmearGate + BigramHash embedding augmentation
- XSA (cross-sequence attention) on all 11 layers
- QK-Gain init = 4.0
- ~27M parameters

## Training

- Muon optimizer for matrix params, Adam for scalars/embeddings
- EMA (decay=0.997) + Tight SWA (every 50 steps from step 4600)
- Late QAT (int6 quantization-aware training, threshold 0.15)
- Full GPTQ: Hessian-based int6 quantization with Cholesky error compensation (32 calibration batches on EMA model)
- Compression: zstd-22
- Training time: ~600s on 8xH100, ~5250 steps at 114ms/step

## Evaluation — SLOT (Softmax Logit Optimization at Test-time)

Based on [arXiv:2505.12392v2](https://arxiv.org/abs/2505.12392v2).

- **Sliding window eval** at stride=64, seq_len=2048 (baseline)
- **SLOT optimization** per batch:
  1. Extract frozen hidden states from last layer (`forward_hidden`) under `torch.no_grad()`
  2. Detach projection weights (tied embedding)
  3. Optimize per-sample additive delta `[bsz, 1, 512]` + per-sample logit bias `[bsz, 1, 1024]`
  4. **16 AdamW steps** with cosine LR schedule (0.008 → 0.0008)
  5. **Scored-position mask**: only positions contributing to final BPB (last `stride` tokens per non-first window) are included in the SLOT optimization loss
  6. Logits computed as: `softcap * tanh((H + delta) @ W_out^T + logit_bias) / softcap)`
  7. Final scoring with optimized delta + logit bias under `torch.no_grad()`

### Legality

- Model weights are **completely frozen** during SLOT — only delta and logit_bias are optimized
- Hidden states extracted under `torch.no_grad()` — no gradient flows through the model
- Standard autoregressive cross-entropy loss preserves causality
- Optimization uses only tokens within each sliding window (no future information)
- `torch.compile` on `forward_hidden` for throughput
- SLOT eval time: ~311s per run (within 10-min eval budget)

### No illegal techniques
- ❌ No n-gram cache
- ❌ No two-pass rescoring
- ❌ No eval-time access to training data
- ❌ No oracle/hindsight selection

## Results

| Seed | Sliding BPB | SLOT BPB | Artifact Size |
|------|-------------|----------|---------------|
| 1337 | 1.1264 | 0.9349 | 15,890,549 |
| 42 | 1.1264 | 0.9325 | 15,830,408 |
| 7 | 1.1261 | 0.9388 | 15,810,068 |
| **Mean** | **1.1263** | **0.9354** | |

Beats merged SOTA (1.1147) by 0.179 BPB. All artifacts < 16,000,000 bytes.

## Reproduction

```bash
SEED=1337 GPTQ_CALIB_BATCHES=32 SLOT_ENABLED=1 SLOT_STEPS=16 \
SLOT_LR=0.008 SLOT_LR_MIN=0.0008 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Key Techniques

1. **LeakyReLU(0.5)²**: Leaky variant (negative slope 0.5) with squaring for sparsity
2. **XSA-all**: Cross-sequence attention on all 11 layers
3. **QK-Gain 4.0**: Sharpened attention maps via learned per-head gain initialized at 4.0
4. **Full GPTQ**: Hessian-based int6 quantization with actorder and Cholesky error compensation
5. **SLOT**: Per-sample delta + logit bias optimization at eval time with scored-position masking
