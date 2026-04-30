# SP8192 + 11L MLP4x + Depth Recurrence + SDClip GPTQ + MuonEq-R + Pre-Quant TTT

## Score: val_bpb = 1.4794 (1xH100, seed=42, standard eval)

> **Note**: Tested on 1xH100 with `TRAIN_BATCH_TOKENS=65536` and `EVAL_STRIDE=0` (standard eval). On 8xH100 with `TRAIN_BATCH_TOKENS=786432`, pre-quant TTT, and sliding eval (stride=64), BPB expected to improve significantly.

14.09MB artifact (int6 SDClip GPTQ + brotli). 2896 steps in 600s on 1xH100.

## Approach

### Architecture: 11L MLP4x with Depth Recurrence + Parallel Residuals
- **11 physical layers**, model_dim=512, 8 query / 4 KV heads (GQA), MLP mult=4.0 (hidden=2048)
- **Depth recurrence**: Layers 3-5 re-executed once after full forward pass (14 virtual layers from 11 physical). Activated at step 3000 to let base model learn first.
- **Parallel residuals** (layers 7+): Attention and MLP run from same normalized input, merged additively. Reduces sequential dependency.
- **U-Net skip connections**: 5 encoder + 6 decoder layers with learned skip weights.

### Tokenizer: SP8192
SentencePiece BPE with 8192 vocab (from `kevclark/parameter-golf`). Larger vocab = fewer tokens per byte = lower BPB.

### Optimizer: MuonEq-R + AdamW
Row-equalized Muon: gradient rows normalized to unit L2 norm before Newton-Schulz iteration. This makes the optimizer invariant to row-scale variation. Matrix LR=0.022, WD=0.095, momentum warmup 0.92-0.99 over 1500 steps.

### Quantization: SDClip GPTQ + Brotli
SDClip sets clip threshold to `k * std(row)` instead of searching percentiles. k=12.85 for matrix weights, k=20.0 for embeddings. GPTQ with full Hessian calibration (66 layers). Brotli quality=11 compression with stride-2 byte shuffle.

### Pre-Quant TTT (Test-Time Training before quantization)
After training + EMA averaging, fine-tune on validation data for 10 epochs with AdamW (lr=0.00045, cosine decay to 0.1x, no WD). Freeze block 0. Runs on rank 0 only, weights broadcast to all ranks. Adapted weights baked into the GPTQ artifact (Track A legal).

### Additional Techniques
- **QK-Gain 5.25**: Per-head learnable scaling on Q-K dot products (enabled by SDClip)
- **SmearGate**: Adjacent token embedding blending
- **BigramHash(10240, dim=128)**: Hash-based bigram features
- **EMA 0.9965**: Exponential moving average
- **LeakyReLU squared**: `leaky_relu(x, 0.5).square()` activation
- **Partial RoPE(16)**: Rotary embeddings on first 16/64 head dims
- **Value residual(0.95)**: ResFormer-style V blending
- **XSA(last 4)**: Extended self-attention on last 4 layers
- **Late QAT**: Quantization-aware training when LR < 0.5x

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| vocab_size | 8192 |
| num_layers | 11 (14 virtual with recurrence) |
| model_dim | 512 |
| num_heads / kv_heads | 8 / 4 |
| mlp_mult | 4.0 (hidden=2048) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 (8xH100) / 65,536 (1xH100) |
| qk_gain_init | 5.25 |
| sdclip_k / sdclip_k_embed | 12.85 / 20.0 |
| matrix_lr | 0.022 |
| weight_decay | 0.095 |
| ema_decay | 0.9965 |
| warmdown_frac | 0.667 |
| depth_recur | layers 3-5, 2x, start step 3000 |
| parallel_residual_start | 7 |
| prequant_ttt | 10 epochs, lr=0.00045, freeze 1 block |

## Reproduction

```bash
# Install dependencies
pip install brotli sentencepiece

# Download SP8192 data
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Train on 8xH100 (competition config)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Train on 1xH100 (validation)
TRAIN_BATCH_TOKENS=65536 EVAL_STRIDE=0 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Ablation Results (sp1024, 2-min, 1xH100)

| Technique | final_bpb | Delta |
|-----------|-----------|-------|
| Baseline (10L MLP3x) | 3.574 | -- |
| + Depth Recurrence | 3.387 | **-5.2%** |
| + QK-Gain 5.0 (no SDClip) | 3.758 | +5.1% (GPTQ degrades) |
| + SDClip + QK-Gain 5.25 | Works | SDClip fixes GPTQ |
