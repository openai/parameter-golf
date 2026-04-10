# Record: 11L + Tight SWA + Shared VE128 + Partial RoPE + LN Scale + XSA4 (val_bpb: 1.1246)

**NEW SOTA** — beats previous record of 1.1248

## Key Innovation: Tight SWA
SWA checkpoint collection restricted to scale<0.2 (last ~600 steps), every 50 steps. This eliminates the SWA quality penalty (post-SWA BPB = pre-SWA BPB) while maintaining quantization-friendly weight averaging. Standard SWA (scale<0.5) averages stale checkpoints that hurt final quality.

## Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared activation
- Efficient Partial XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Shared Value Embedding (dim=128, layers 9,10) — 1 table, per-layer learned scales
- FlashAttention 3 (Hopper)
- Orthogonal init with proj scaling by 1/sqrt(2*num_layers)
- Logit softcap 30.0, tied embeddings

## Training
- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3000 iters (wallclock-based)
- **Tight SWA**: every 50 steps when scale<0.2 (12 checkpoints from last 600 steps)
- Late QAT: STE int6 fake-quantization when LR scale<0.1

## Quantization
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

## Results
- 6942 steps in 600s at 86.4ms/step
- Pre-quant val_bpb: 1.1407
- Post-SWA val_bpb: 1.1407 (zero SWA penalty!)
- Quant gap: 0.008
- **Sliding window val_bpb: 1.1246**
- Artifact size: 15,706,024 bytes (15.71 MB)

## Run
```bash
NUM_LAYERS=11 MLP_MULT=3.0 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
SWA_ENABLED=1 SWA_EVERY=50 LATE_QAT_THRESHOLD=0.1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 ADAM_WD=0.04 MUON_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
torchrun --nproc_per_node=8 train_gpt.py
```
