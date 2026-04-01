# 11L Int4-MLP QAT + BigramHash(10240) + SWA(frac=0.4)

**val_bpb: TBD** (pending 3-seed evaluation on 8xH100)

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=42)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

## Key Techniques

### Int4 MLP with Quantization-Aware Training (QAT)
- **Int4 [-8,7]** for MLP weights via STE fake quantization during training
- **Int6 [-32,31]** for attention weights (precision-sensitive, post-training only)
- **FP16** for tied embeddings and last-layer key projections
- Int4 MLP saves ~2MB vs int5, funding an 11th layer within the 16MB budget
- QAT trains the model to be robust to int4 precision loss

### 11 Transformer Layers
- Extra layer enabled by int4 MLP compression savings
- Expected ~0.004-0.008 bpb improvement from additional capacity
- U-Net skip connections: encoder (5 layers) + decoder (6 layers)

### All SOTA Innovations Preserved
- BigramHash(10240, dim=128) with learned projection
- SmearGate for bigram context at embedding layer
- Orthogonal init with muP-scaled output projections
- SWA (start_frac=0.4, every 50 steps)
- 5% magnitude pruning for MLP, 3% for attention
- zstd-22 compression
- Sliding window eval (stride=64)

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- U-Net skip connections, tied embeddings

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=2500 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, int4 QAT for MLP layers
- SWA: start_frac=0.4, every=50 steps
- Sliding window eval: stride=64

Built on SOTA submission by @thwu1 (Int5-MLP + BigramHash 10240).
