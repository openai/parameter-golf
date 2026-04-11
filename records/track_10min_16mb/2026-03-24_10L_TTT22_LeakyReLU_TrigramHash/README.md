# 10L TTT 22ep AdamW Cosine + LeakyReLU(0.5)² + TrigramHash

**val_bpb: 1.1354** | **artifact: 15.35 MB** | **8xH100 SXM, 600s train + 603s eval**

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA), MLP 3x
- RoPE, tied embeddings, logit softcap 30.0
- Value Residual (ResFormer), Gated Attention, XSA last 4 layers
- LeakyReLU(0.5)² activation (preserves negative gradient flow)
- TrigramHash + BigramHash (shared 2048-bucket embedding table)
- SmearGate, LN Scale (depth-scaled residuals)
- U-Net skip connections

## Training
- Muon optimizer (Newton-Schulz) for matrices, AdamW for embeddings/scalars
- MATRIX_LR=0.03, warmdown 3500 iters
- SWA: 27 checkpoints averaged
- Late QAT: threshold 0.5, STE fake-quantization during warmdown
- 5195 steps in 600s on 8xH100

## Quantization
- Mixed int5 (MLP) / int6 (attention) with GPTQ-lite per-row clip search
- 3% magnitude pruning before quantization
- FP16 passthrough for embeddings + control tensors
- zstd-22 compression

## Test-Time Training
- 22 epochs AdamW (lr=0.0005, wd=0.0)
- Per-step cosine LR decay to 0
- Per-layer LR groups: 3x for output projections, 0.5x for input projections
- Batched 32 sequences per GPU, distributed gradient sync via all_reduce
- Gradient clipping at 1.0
- TTT time: 406s, eval time: 197s (total: 603s)

## Key Findings
- Batched TTT (32 seqs/GPU) is ~500x faster than chunk-based (1 seq × 256 tokens)
- Per-step cosine decay prevents overfitting at high epoch counts
- Gradient sync per step (not post-TTT) is critical for multi-GPU TTT
- LeakyReLU(0.5)² gives -0.003 BPB over ReLU²
- TrigramHash extends BigramHash context from 2 to 3 tokens using shared embedding table
