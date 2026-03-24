# Record: 11L XSA-all + Full GPTQ + Parallel Muon + Selective Pruning

**val_bpb: 1.1171** (3-seed mean, std 0.0006) | **15.92 MB** max artifact | 8xH100 SXM, 600s

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s64) | val_loss | Artifact |
|------|-------|---------|--------------------|----------|----------|
| 1337 | ~7,100 | 84.2 | **1.1164** | 1.8851 | 15,920,050 bytes |
| 42 | ~7,100 | 84.2 | 1.1171 | 1.8861 | 15,921,954 bytes |
| 7 | ~7,100 | 84.2 | 1.1177 | 1.8871 | 15,914,654 bytes |

**Mean: 1.1171 | Std: 0.0006**

## Key Techniques

### XSA on All 11 Layers
Standard practice applies Exclusive Self-Attention to only the last 4 layers. Applying to all 11 forces cross-position information mixing from layer 0, improving representation quality. Zero new parameters — just a config change. -0.0016 BPB vs XSA-last-4 in ablation.

### Full Hessian GPTQ with amax-aligned QAT
- 256-sample calibration from training data for per-layer Hessian approximation
- Column-wise int6 quantization with Cholesky error compensation, block size 128
- QAT STE aligned to export quantizer using row-maximum (amax) clipping with [-32, 31] range
- Late QAT at threshold 0.15

### Parallel Muon Optimizer with Parameter Banking
- Weight matrices stored in contiguous parameter banks (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)
- 3-phase overlapped optimizer step: async reduce-scatter → batched Newton-Schulz orthogonalization → async all-gather
- Eliminates DDP double-communication overhead, achieving 84.2ms/step (~7,100 steps in 600s)

### Selective ±1 Magnitude Pruning
Post-GPTQ, sort quantized values at ±1 by reconstruction error (scale²), zero least-impactful first until artifact fits target. Binary search for exact target size. Targets only values whose removal causes minimal reconstruction damage.

### LZMA Compression
LZMA preset 6 replacing zstd-22 for model serialization. Better compression ratio on int6 quantized weights.

## Architecture

- 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (hidden=1536) with **LeakyReLU(0.5)²** activation
- **XSA on all 11 layers** (Exclusive Self-Attention)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate temporal gating
- BigramHash (2048 buckets, 128-dim)
- Shared Value Embedding (dim=128, layers 9-10)
- FlashAttention 3 (Hopper native kernels)
- Orthogonal init, logit softcap 30, tied embeddings

## Training

- Parallel Muon optimizer (matrices): lr=0.025, momentum=0.99, WD=0.04, 5 Newton-Schulz steps
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3,500 iters (wallclock-based)
- EMA (decay=0.997) + Tight SWA (every 50 steps, scale<0.2)
- Late QAT: STE int6 fake-quantization when LR scale<0.15

## Quantization & Compression

- Full GPTQ with 256-sample Hessian calibration, block_size=128, percdamp=0.01
- Int6 per-row with amax clipping, range [-32, 31]
- Selective ±1 magnitude pruning (target 15.9MB)
- Small tensors + tok_emb.weight in fp16
- LZMA preset 6 compression

## Requirements

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install zstandard sentencepiece
```

## Run Command

```bash
SEED=1337 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- [x] 3 seeds run on 8xH100 SXM
- [x] All 3 seeds train in ≤600s
- [x] All 3 seeds artifact ≤16,000,000 bytes (max: 15,921,954)
- [x] Sliding window eval stride=64, consistent (std=0.0006)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
