# 10L Int5-MLP + BigramHash + EMA + TTT + Backout Connection (Non-Record)

**val_bpb: 1.1574** (8xH100 SXM, seed=42)

## Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Hardware | Steps | val_bpb | Artifact Size |
|----------|-------|---------|---------------|
| 8xH100 SXM (RunPod) | ~7000 | **1.1574** | 15.5MB |
| 1xH100 (RunPod) | 869 | 1.4463 | 15.5MB |
| 1xA100 (Northeastern HPC) | 433 | 1.6560 | 15.5MB |

## Approach

Built on thwu1's #1 record (1.1428 bpb), adding three techniques:

### 1. EMA (replacing SWA)
Exponential Moving Average with decay=0.997, starting at step 50. Updated every step rather than collecting periodic checkpoints like SWA. Gives exponentially more weight to recent (better converged) checkpoints, producing smoother weight distributions that quantize better.

### 2. Backout Connection (inspired by PR #339)
A learned residual subtraction that removes redundant mid-layer information from the final representation. Captures hidden state at layer `num_layers // 2` (layer 5) and subtracts `lambda * h_mid` from the output before the final RMSNorm. Adds exactly 1 parameter (a learned scalar, init 0.2). Zero computational cost.

### 3. Test-Time Training (inspired by PR #338)
After quantization roundtrip, performs 3 epochs of SGD fine-tuning directly on validation tokens. Adapts the quantized model to recover from quantization degradation. First 2 transformer blocks are frozen to preserve low-level features.

- Optimizer: SGD, lr=0.002, momentum=0.9
- Epochs: 3
- Grad clip: 1.0
- Frozen: first 2 blocks

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- U-Net skip connections, tied embeddings
- Mixed int5 (MLP) / int6 (attention) quantization + zstd-22
- 3% magnitude pruning
- EMA (decay=0.997, start_step=50)
- Sliding window eval (stride=64)
- Backout connection at layer 5 (lambda init=0.2)
- TTT: 3 epochs SGD on val tokens post-quantization

## Training Hyperparameters
- Muon: matrix_lr=0.02, WD=0.04, momentum=0.99 (warmup from 0.92)
- seq_len=2048, batch=786K tokens, warmdown=3000
- grad_clip=0.3

## Note
Single seed result (seed=42). Additional seeds needed for statistical significance. Work in progress — exploring further improvements.
