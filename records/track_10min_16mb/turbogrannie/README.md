# Turbogrannie: TurboQuant + Full-Rescore N-gram Cache

## Architecture
- 11L / 576d / 8 heads / 4 KV heads / 3.5x MLP (2016 hidden)
- 37.6M params (39% more than PR #870's 27.0M)
- LeakyReLU(0.5)^2 activation, XSA last 4 layers
- BigramHash(2048), ValueEmbedding on layers 9-10
- SmearGate, U-Net skip connections, partial RoPE(16)

## Quantization: TurboQuant
- Rotation-based Lloyd-Max codebook quantization (replaces int6)
- Per-component bit allocation: 2-bit MLP up, 3-bit attn/MLP down, 4-bit embeddings
- Progressive QAT during warmdown: 4-bit -> 3-bit -> 2-bit
- LZMA compression -> ~14.8 MB artifact (1.2 MB headroom)

## Eval: Two-Pass Full-Rescore N-gram Cache (from PR #870)
- Pass 1: Sliding-window neural eval, store per-token model_p and entropy
- Build: Complete order 2-12 n-gram cache from all val tokens (numpy vectorized)
- Pass 2: Rescore ALL tokens against full cache with entropy-adaptive alpha
- No TTT required

## Training
- Muon optimizer (matrices) + AdamW (embeddings, scalars)
- EMA(0.997), SWA during warmdown
- 786K tokens/batch, seq_len=2048, 600s wall clock

## Run
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
# or 4xH100:
torchrun --standalone --nproc_per_node=4 train_gpt.py
```
