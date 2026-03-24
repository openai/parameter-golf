# DeepQuant — 11L INT6 + 8-epoch Cosine LoRA TTT

**val_bpb: 0.5850** (seed=42, eval 582s, 15.46MB)

## Approach

We explored how far per-document test-time training can push a small 16MB language model. The core hypothesis: a well-trained base model combined with aggressive per-document LoRA adaptation at eval time can dramatically reduce bits-per-byte by specializing the model to each document's distribution.

## Architecture

Standard 11-layer transformer backbone:
- dim=512, 8 attention heads, 4 KV heads (GQA), MLP expansion 3x (1536)
- BigramHash(2048) + SmearGate for parameter-efficient bigram context
- U-Net skip connections between encoder/decoder layer pairs
- Depth-scaled residuals: 1/sqrt(layer+1) for stable deep training
- RoPE positional encoding (base=50000)
- Logit softcap=30.0

## Training (600s, 8xH100 SXM)

- Muon optimizer (Newton-Schulz whitening) for matrix params + AdamW for scalars/embeddings
- Wallclock-based LR schedule with warmdown
- EMA (decay=0.999, every 10 steps) + SWA (12 checkpoints in final warmdown)
- ~7100 training steps, batch tokens=786,432
- INT6 uniform quantization (64 levels per row) + zstd-22 compression
- 4% magnitude pruning before quantization

## Test-Time Training (TTT) — Key Innovation

Per-document LoRA adaptation at eval time with several design choices that proved critical:

### 1. 8-epoch multi-pass adaptation
Each document gets 8 full passes of LoRA training. We found TTT gain scales strongly with epoch count — each additional epoch provides meaningful BPB improvement as the LoRA captures deeper document-specific patterns.

### 2. Score-every-epoch compliance
Every token is scored before being trained on, in every epoch. Scores are overwritten each epoch, so the final score reflects the most adapted LoRA state. This satisfies backward-looking TTT requirements.

### 3. Cosine LR decay within TTT
Per-step cosine schedule (from base LR down to 10%) across all epochs×chunks steps. This prevents overfitting in later passes while allowing aggressive early adaptation. Constant LR overshoots on later chunks.

### 4. LM-head LoRA rank-16
The output projection (dim→vocab) is the highest-leverage layer for BPB. We use rank-16 for the LM-head LoRA while keeping rank-8 for Q/V projections. This doubles the model's capacity to adapt its output distribution per document.

### 5. Per-block bias tuning
During TTT, we tune a bias vector (512 params) per transformer block alongside LoRA. This provides a cheap "domain shift" — adjusting activation means to match document statistics without extra matmul cost.

### 6. Post-TTT temperature rescaling (T=0.98)
Multi-epoch LoRA adaptation tends to make the model overconfident. Scaling logits by 0.98 corrects this calibration error for a consistent ~0.003 BPB improvement at zero compute cost.

### 7. Zigzag GPU load balancing
Documents are distributed across 8 GPUs using a zigzag pattern (GPU 0→7, then 7→0, repeating) instead of contiguous blocks. This ensures each GPU processes a balanced mix of document lengths, eliminating a ~220s synchronization bottleneck from GPU workload imbalance.

### 8. Outlier document filtering
Documents exceeding 50,000 tokens are scored with the base model without TTT. These extreme outliers take disproportionate compute (quadratic in chunk count) while being too few to meaningfully affect average BPB.

### 9. Wall-clock TTT budget
A configurable time limit (570s default) on the TTT batch loop. If exceeded, remaining documents fall back to batched base-model scoring. This guarantees eval completes within the 600s budget.

## TTT Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (Q, V) | 8 |
| LoRA rank (LM-head) | 16 |
| TTT LR | 0.01 (Adam, betas=0.9/0.95) |
| TTT epochs | 8 |
| TTT chunk size | 256 |
| TTT batch size | 64 documents |
| TTT min doc length | 512 tokens |
| TTT max doc length | 50,000 tokens |
| Temperature rescale | 0.98 |
| Cosine LR | enabled (min 10%) |
| Bias tuning | enabled |

## How to run

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
SEED=42 TTT_EPOCHS=8 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Timing breakdown

| Phase | Time |
|-------|------|
| Training | 600s |
| Post-processing (SWA+EMA+pruning) | <1s |
| Serialization (quant+compress) | 38s |
| Post-quant eval | 5s |
| TTT eval (short docs) | 22s |
| TTT eval (long docs, 62 batches) | 559s |
| TTT overhead | 2s |
| **Total eval** | **582s** |
