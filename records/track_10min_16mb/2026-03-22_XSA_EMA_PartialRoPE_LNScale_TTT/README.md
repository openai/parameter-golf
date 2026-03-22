# XSA + EMA + Partial RoPE + LN Scale + TTT

**val_bpb: 1.1365** (seed 42, 8xH100 SXM, 600s)

## Architecture
- 10 transformer layers, 512d, 8 heads / 4 KV heads (GQA), 3x MLP (1536 hidden)
- Exclusive Self-Attention (XSA) on last 4 layers
- EMA (decay=0.997) replacing SWA
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer_idx+1))
- SmearGate + BigramHash(10240, dim=128)
- Backout: subtract mid-layer residual before logit head
- LeakyReLU(0.5)^2 activation
- U-Net skip connections, orthogonal init, logit softcap=30

## Training
- Muon optimizer with cautious weight decay (WD=0.04, momentum 0.99)
- AdamW for embeddings/scalars
- Batch tokens: 786,432, seq len: 2048
- Warmdown: 3000 iters
- 6491 steps in 600s (92.4ms/step)

## Quantization
- Int5 MLP / Int6 attention per-row quantization
- FP16 embedding passthrough, FP16 last-layer c_k
- 3.2% magnitude pruning
- zstd level 22 compression
- Artifact: 15,759,319 bytes (code: 55,693)

## Evaluation
- Sliding window (stride=64, seq_len=2048)
- TTT: 3 epochs SGD (lr=0.002, momentum=0.9) on val data post-quantization
- First 2 blocks frozen during TTT

## Base
Built on thwu1's 10L Int5-MLP submission (1.1428 BPB) with zero-parameter architectural improvements (XSA, EMA, Partial RoPE, LN Scale) and techniques from nanochat.

## Environment
- PyTorch 2.7.0 + FlashAttention 2.8.3
- 8xH100 SXM 80GB
