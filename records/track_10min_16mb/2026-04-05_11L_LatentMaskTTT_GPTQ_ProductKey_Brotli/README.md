# 11L LatentMask TTT + GPTQ + Product-Key Bigram + Brotli

**val_bpb: 1.1158 (3-seed mean)** | Artifact: ≤15,989,386 bytes | 8xH100, 600s training + ~457s eval

## Summary

11-layer GPT with U-Net skip connections, achieving 1.1158 val_bpb (3-seed mean) through four key innovations:

1. **LatentMask TTT (Test-Time Training)**: Per-channel sigmoid masks + biases on MLP and attention outputs, trained per-chunk during evaluation using a sign-based Muon-lite optimizer. Score-first (legal): each chunk is scored before any mask update. Provides ~−0.002 bpb improvement over sliding window baseline.

2. **Full Hessian GPTQ**: Hessian-aware int6 quantization with Cholesky error compensation and column reordering. Uses autoregressive self-generated calibration data (32 sequences × 2048 tokens). Reduces quantization error vs per-row percentile search.

3. **Product-Key Bigram Embedding**: Factored bigram via `embed_prev(1024,512) * embed_cur(1024,512)` — zero hash collision, no projection layer needed. Replaces traditional hash-based bigram embedding.

4. **Brotli-11 Compression**: Custom binary serialization (JSON header + raw tensor bytes) compressed with Brotli quality=11. Combined with uint8 log-scale quantization for per-row scales.

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x (1536 hidden), LeakyReLU(0.5)²
- GatedAttention on even layers [0,2,4,6,8,10]
- XSA (Exclusive Self-Attention) on all 11 layers
- Value Embeddings at decoder layers [5,7,10]
- U-Net encoder-decoder skip connections
- SmearGate for adjacent token mixing
- Tied embeddings, logit softcap=30

## Training

- Muon optimizer (matrix params), AdamW (scalar/embed params)
- matrix_lr=0.028, muon_wd=0.0417
- EMA (decay=0.997), Late QAT (threshold=0.15)
- Warmdown: 3500 steps (time-based adaptive)
- ~5,830 steps in 600s, step_avg ~102.9ms (H100)

## Quantization & Compression

- int6 per-row for MLP/attention weights (GPTQ when Hessian available)
- int8 per-row for embeddings
- uint8 log-scale for per-row scales (2B → 1B per scale)
- Custom binary serialization + Brotli-11

## Evaluation

- LatentMask TTT: lr=0.0008, chunk=65536 tokens, epochs=4, momentum=0.9
- Sliding window stride=64, seq_len=2048
- TTT eval time: ~457s (H100)

## Results (3-seed, 8xH100)

| Seed | val_bpb | Steps | Step Avg (ms) | Artifact (bytes) |
|------|---------|-------|---------------|-------------------|
| 777  | 1.11561 | 5,869 | 102.24 | 15,980,910 |
| 999  | 1.11594 | 5,808 | 103.31 | 15,989,386 |
| 1337 | 1.11587 | 5,814 | 103.21 | 15,977,334 |
| **Mean** | **1.11581** | **5,830** | **102.92** | |

## Dependencies

```
pip install flash-attn brotli
```

(`sentencepiece`, `torch`, `numpy` assumed pre-installed)

## Run

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
