# 11L U-Net + Catalytic Residuals + SwiGLU + Sliding Window

**val_bpb: 1.1558** (1 seed)

## Key Techniques

1. **11 transformer layers** with U-Net gated skip connections: Sigmoid-gated blending between encoder and decoder layers (`gate * x + (1-gate) * (weight * skip)`). 5 encoder + 1 mid + 5 decoder layers.

2. **Catalytic residuals** (PR #450): Learned per-dim gates on attention and MLP outputs, initialized to 1.0. Zero compute overhead, ~0.024 BPB improvement.

3. **SwiGLU MLP**: Gated linear unit with SiLU activation, 3× expansion factor.

4. **Value residual** (ResFormer): Blend first-layer V into all subsequent layers for better gradient flow.

5. **Sliding window evaluation** (stride=64, seq_len=1024): Every token scored with 960+ context.

6. **LN scale dampening**: `1/sqrt(layer_idx+1)` on RMSNorm inputs for deeper layers.

7. **Decoder LR multiplier** (2×): Decoder layers get higher learning rate for both Muon and Adam.

8. **Int5/Int6 mixed quantization + zstd-22**: MLP and bigram weights at 5-bit, rest at 6-bit.

9. **BigramHash** (4096 buckets, 128-dim): Bigram-conditioned token embeddings via hash-based lookup.

10. **EMA** (decay=0.9985): Exponential moving average of weights.

## Architecture

- 11 blocks: dim=512, 8 attn heads, 4 KV heads (GQA), 3× MLP
- Vocab: 1024 (SentencePiece BPE), tied embeddings
- Seq len: 2048 (train), 1024 (eval sliding window)
- Partial RoPE (25% of head dims)
- XSA on last 4 layers, gated attention

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.9516 | 1.1558 | 6898 | 87.0 |

Pre-quant EMA: val_bpb=1.1606 | Post-quant: val_bpb=1.1723 | Sliding window: val_bpb=1.1558

Artifact: 15.1 MB (15,192,709 bytes) | Eval time: ~172s (sliding window)

## Training

- Muon optimizer (momentum=0.99, WD=0.04) + Adam (scalar params)
- 524,288 tokens/step, 6,898 steps in 600s on 8×H100 SXM
- Warmdown: 3000 iters, grad clip: 0.3
