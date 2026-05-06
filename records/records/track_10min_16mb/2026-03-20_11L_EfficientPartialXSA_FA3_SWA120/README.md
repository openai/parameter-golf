# 11L + Efficient Partial XSA (val_bpb: 1.1307)

## Results
- **val_bpb: 1.1307** (sliding window, stride=64)
- Pre-quantization BPB: 1.1437
- Model parameters: 26,829,913
- Artifact size: 15,892,986 bytes (under 16MB limit)
- Training: 6,976 steps in 600 seconds (~86ms/step)
- SWA: 13 checkpoint average during warmdown (every 120 steps)

## Novel Contribution: Efficient Partial Exclusive Self Attention (XSA)

Based on Exclusive Self Attention (arXiv:2603.09078), we introduce two key improvements:

### 1. Efficient GQA-Aware Implementation
Standard XSA with Grouped Query Attention requires `repeat_interleave` to expand value vectors
from `num_kv_heads` to `num_heads`, doubling memory allocation per layer. Our implementation
uses a free reshape into KV head groups + broadcasting:

```python
# OLD: expensive tensor duplication
v_expanded = v.repeat_interleave(group_size, dim=-2)  # allocates 2x memory
vn = normalize(v_expanded)
y = y - dot(y, vn) * vn

# NEW: free reshape + broadcast (zero allocation)
y_grouped = y.reshape(B, T, Hkv, group_size, D)      # view, no copy
vn = normalize(v).unsqueeze(-2)                        # [B,T,Hkv,1,D]
y = (y_grouped - dot(y_grouped, vn) * vn).reshape(B, T, H, D)
```

This reduces XSA overhead from ~7ms/step to ~2ms/step at 11 layers with GQA (8 heads, 4 KV heads).

### 2. Partial Application to Deepest Layers Only
The XSA paper shows self-attention bias (cosine similarity between output and self-value)
increases across layers. We apply XSA only to the **last 3 layers** (out of 11), targeting
the layers with highest self-attention bias while minimizing compute overhead.

Combined, these give ~0.002 BPB improvement over the baseline at <2ms/step cost.

## Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads via GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (encoder=5, decoder=6)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Tied embeddings, logit softcap=30.0
- NTK-aware RoPE (train_seq_len=1024, auto-scales at 2048)
- **XSA on layers 8, 9, 10** (deepest 3 of 11)

## Training
- FlashAttention 3 (Hopper-optimized)
- Muon optimizer: lr=0.025, momentum=0.99 (warmup from 0.92 over 1500 steps)
- AdamW for embeddings/scalars: lr=0.035/0.025
- Weight decay: 0.04 (both Muon and AdamW)
- Warmdown: 3000 iterations, grad clip 0.3
- SWA every 120 steps (scale < 0.5), 13 checkpoint uniform average
- OrthoInit + muP-scaled output projections
- Seed: 1337

## Quantization
- Int6 per-row quantization on MLP + attention weights
- Int8 for embeddings
- zstd level 22 compression

## Run Command
```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SWA_EVERY=120 SWA_ENABLED=1 MTP_NUM_HEADS=0 SEED=1337 \
WARMUP_STEPS=30 VAL_LOSS_EVERY=2000 XSA_LAST_N=3 \
torchrun --nproc_per_node=8 train_gpt.py
```

## References
- Exclusive Self Attention: arXiv:2603.09078 (Shuangfei Zhai, 2026)
