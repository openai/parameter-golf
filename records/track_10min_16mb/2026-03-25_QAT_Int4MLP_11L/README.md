# 11L QAT Int4-MLP + Int6-Attn + BigramHash(4096) + SmearGate + SWA

**val_bpb: TBD** (post-quant int4/int6+zstd, sliding window stride=32)

## Key Changes from SOTA (#1, 1.14276 bpb)

### 1. Quantization-Aware Training (QAT) via STE
Every forward pass fake-quantizes weights to their target precision using Straight-Through Estimator. The model learns weight distributions inherently robust to post-training quantization, reducing the quantization penalty.

- MLP weights: int4 (clip_range=7, [-8,7]) during training
- Attention weights: int6 (clip_range=31, [-32,31]) during training
- Embeddings: FP16 passthrough (no QAT)

### 2. Int4 MLP Quantization (was Int5)
MLP weights quantized to int4 instead of int5. QAT ensures the model adapts to the coarser grid. Int4 compresses ~2.5-3x with zstd-22 (vs ~1.88x for int5), saving enough bytes to fund an 11th transformer layer.

### 3. 11th Transformer Layer (was 10)
The byte savings from int4 MLP + smaller bigram table fund an extra layer. More depth improves the model's representational capacity.

### 4. Reduced BigramHash (4096 vs 10240 buckets)
Traded bigram resolution for an extra layer. The SOTA ablation showed 10240->4096 costs only ~0.0008 bpb, while an extra layer should recover more.

### 5. Sliding Window Eval at stride=32 (was 64)
Halved eval stride for more overlap. Each scored token gets ~2016 tokens of context. Doubles eval time but should still be well under the 10-minute eval limit.

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(4096, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- QAT: int4 MLP (clip=7), int6 attention (clip=31)

## Run Command

```bash
RUN_ID=qat_int4mlp_11L \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Improvements
- QAT reduces quantization penalty (est. -0.001 to -0.005 bpb)
- 11th layer adds capacity (est. -0.003 bpb based on 9L->10L gain)
- stride=32 eval (est. -0.001 bpb)
- Net expected: -0.003 to -0.007 bpb vs 1.14276 SOTA

Built on the #1 submission by thwu1 (10L Int5-MLP + BigramHash(10240) + SWA).
