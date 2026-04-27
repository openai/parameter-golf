# 9L GPTQ-lite Int8 + BigramHash(2048) + EMA(0.9999)
val_bpb: 1.3058 (seed 1337, sliding window, post int8+zlib quantization roundtrip)

## Run Command
```bash
python train_gpt.py
```
All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Results
| Seed | val_bpb | artifact_bytes | valid |
|------|---------|----------------|-------|
| 1337 | 1.3058  | 14,719,767     | yes   |

## Key Techniques

- **GPTQ-lite Int8 Quantization**: Modifies standard quantization by searching 5 candidate clip percentiles (0.999 to 1.0) per row and picking the one that minimizes reconstruction Mean Squared Error (MSE).
- **BigramHash(2048)**: Hashes consecutive token pairs into a 2048-bucket embedding table (hidden dim 6144), which is then projected to `model_dim=512` via a learned linear layer. This reduces token-pair hash collisions and enhances context awareness early in the model.
- **Late QAT (Quantization-Aware Training)**: Applies Straight-Through Estimator (STE) int6 fake-quantization during the final 15% of training steps (`qat_threshold = 0.15`). Prepares weights for the precision drop during post-training quantization.
- **EMA (Exponential Moving Average)**: Maintains a shadow copy of model parameters with `decay=0.9999` throughout training. Final quantization occurs on the EMA weights rather than raw weights, resulting in a smoother, more robust evaluation model.
- **LeakyReLU² MLP**: Uses `(leaky_relu(x, slope=0.5))²` instead of `ReLU²`, allowing small negative gradients to flow during early steps to completely avoid dead neurons.

## Architecture
- 9 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 2x expansion, LeakyReLU² activation
- Standard Token Embedding + BigramHash Embedding
- U-Net style skip connections (skips from the first half of layers are added to the later half)
- Tied embedding mappings

## Training Hyperparameters
- **Optimizers**: Muon for 2D matrix parameters (`lr=0.04`, momentum=0.95) and Adam for embeddings/scalars.
- **Schedule**: 20,000 iterations total, 20 warmup steps, and 1,200 warmdown iters.
- **Batching**: `seq_len=1024`, `batch=524K` tokens per step.
- **Regularization**: `grad_clip=0.0` (disabled by default in script). EMA applied every step.