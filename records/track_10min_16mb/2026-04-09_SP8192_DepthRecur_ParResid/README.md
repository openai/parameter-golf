# SP8192 + Depth Recurrence + Parallel Residuals

## Score: val_bpb = 1.6323 (1xH100, seed=42)

> **Note**: This result was obtained on **1xH100** (1/8 of the competition's 8xH100 compute). With 8xH100 and `TRAIN_BATCH_TOKENS=786432`, the BPB will improve substantially due to increased effective batch size and data throughput.

Trained on 1xH100 SXM 80GB in 600 seconds. 14.09MB artifact (int6+gptq+zlib).

## Approach

### 1. SP8192 Tokenizer
SentencePiece BPE with 8192 vocab (from `kevclark/parameter-golf` on HuggingFace). Larger vocabulary reduces tokens-per-byte, directly lowering BPB. The 8192-token embedding table (8192x512 = 4.2M params) fills more of the 16MB artifact budget compared to the default 1024-vocab (0.5M params).

### 2. Depth Recurrence (layers 3-5, 2x)
After the normal forward pass through all 10 layers (with U-Net skip connections), layers 3-5 are re-executed a second time. This gives 13 effective transformer layers from only 10 physical layers -- more compute per parameter without increasing model size. Depth recurrence improved final BPB by 5.2% in our ablation studies.

### 3. Parallel Residuals (layers 7+)
On layers 7-9, attention and MLP run in parallel from the same normalized input:
```python
x = x + attn(norm(x)) + mlp(norm(x))  # parallel
```
instead of sequential `x = x + attn(norm(x)); x = x + mlp(norm(x + attn_out))`. This reduces sequential dependency and empirically improves quality on deeper effective models.

### 4. U-Net Skip Connections
The 10-layer model is split into 5 encoder + 5 decoder layers. Encoder outputs are connected to decoder inputs via learned skip weights, enabling bidirectional information flow.

### 5. Full-Hessian GPTQ with Percentile-Search Scales
GPTQ quantization uses calibration Hessians from 61 model layers. Scale factors are optimized by searching over 5 percentile clip values (0.999, 0.9995, 0.9999, 0.99999, 1.0), selecting the one that minimizes per-row MSE. MLP weights use int5 (clip=15), attention weights use int6 (clip=31).

### 6. Additional Techniques
- **BigramHash(10240, dim=128)**: Hash-based bigram embedding table
- **SmearGate**: Learned gate blending adjacent token embeddings
- **EMA(0.997)**: Exponential moving average of weights
- **LeakyReLU^2**: `leaky_relu(x, 0.5).square()` activation in MLP
- **GQA(8q/4kv)**: Grouped-query attention
- **Partial RoPE(16)**: Rotary position embeddings on first 16 dims only
- **Value Residual(0.95)**: ResFormer-style blending of current and initial value projections
- **XSA(last 4)**: Extended self-attention on last 4 layers
- **Orthogonal Init**: With `1/sqrt(2*num_layers)` scaling on output projections
- **Muon + AdamW**: Muon for matrix params, AdamW for embeddings/scalars
- **Late QAT**: Quantization-aware training activated when LR < 0.5x

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| vocab_size | 8192 |
| num_layers | 10 (13 effective with depth recurrence) |
| model_dim | 512 |
| num_heads | 8 (4 KV heads) |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 1024 |
| train_batch_tokens | 65536 (1xH100) / 786432 (8xH100) |
| max_wallclock_seconds | 600 |
| depth_recur | layers 3-5, count 2 |
| parallel_residual_start | 7 |
| qk_gain_init | 1.5 |
| ema_decay | 0.997 |
| warmdown_iters | 3500 |
| weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| late_qat_threshold | 0.50 |

## Reproduction

```bash
# Download sp8192 tokenizer and data
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# Train (8xH100)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Train (1xH100, for validation)
TRAIN_BATCH_TOKENS=65536 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Ablation Results (1xH100, sp1024, 2-min runs)

| Technique | Steps | final_bpb | Delta |
|-----------|-------|-----------|-------|
| Baseline | 414 | 3.574 | -- |
| + Depth Recurrence | 339 | 3.387 | -5.2% |
| + Big Batch (262K) | 174 | 3.448 | -3.5% |
| + QK-Gain 5.0 | 457 | 3.758 | +5.1% (GPTQ degrades) |
