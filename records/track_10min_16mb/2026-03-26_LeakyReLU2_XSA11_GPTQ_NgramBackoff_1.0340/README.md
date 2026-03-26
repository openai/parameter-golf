# LeakyReLU² + XSA-all + Full GPTQ + 5-gram Backoff

**val_bpb: 1.0340** (3-seed mean: 1337→1.0342, 42→1.0340, 7→1.0338)

## Architecture

- 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)² MLP with 2x expansion
- RoPE, RMSNorm, tied embeddings (vocab=1024), logit softcapping (30.0)
- U-Net skip connections with learned skip weights
- SmearGate + BigramHash embedding augmentation
- XSA (cross-sequence attention) on all 11 layers
- ~27M parameters

## Training

- Muon optimizer for matrix params, Adam for scalars/embeddings
- EMA (decay=0.997) + Tight SWA
- Late QAT (int6 quantization-aware training)
- Full GPTQ: Hessian-based int6 quantization with Cholesky error compensation (32 calibration batches on EMA model)
- Compression: zstd-22
- Training time: ~600s on 8xH100, ~5250 steps at 114ms/step

## Evaluation

- Sliding window eval at stride=64, seq_len=2048
- **5-gram multi-order backoff**: cascade 5→4→3→2-gram lookup with separate hash tables per order
- **Entropy-adaptive alpha**: alpha = 0.05 + 0.35 * sigmoid(2*(H-4.0)), where H is model entropy
  - Low entropy (confident model): alpha ≈ 0.05, trust model
  - High entropy (uncertain model): alpha ≈ 0.40, trust n-gram cache
- Score-first protocol: each token scored before its n-gram is added to cache
- Hash tables: 4M buckets per order, uint32 counts, min_count=2

## Results

| Seed | Sliding BPB | N-gram BPB |
|------|-------------|------------|
| 1337 | 1.1273 | 1.0342 |
| 42 | 1.1272 | 1.0340 |
| 7 | 1.1269 | 1.0338 |
| **Mean** | **1.1271** | **1.0340** |

Artifact size: 15,903,061 bytes (< 16,000,000)

## Reproduction

```bash
SEED=1337 GPTQ_CALIB_BATCHES=32 \
NGRAM_EVAL_ORDER=5 NGRAM_BACKOFF=1 NGRAM_ENTROPY_ADAPTIVE=1 \
NGRAM_ALPHA_LOW=0.05 NGRAM_ALPHA_HIGH=0.40 NGRAM_ENTROPY_THRESH=4.0 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Key Techniques

1. **LeakyReLU(0.5)²**: Replaces relu² with leaky variant (negative slope 0.5), providing better gradient flow while maintaining sparsity from squaring
2. **XSA-all**: Extended cross-sequence attention from last 4 layers to all 11
3. **Full GPTQ**: Hessian-based quantization with actorder and Cholesky error compensation, calibrated on training data within the training budget
4. **N-gram backoff**: Multi-order cascade (5→4→3→2) with separate tables per order, using entropy-adaptive mixing weights
