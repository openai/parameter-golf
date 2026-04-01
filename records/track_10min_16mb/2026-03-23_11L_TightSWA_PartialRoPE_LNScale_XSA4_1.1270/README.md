# Record: 11L + Tight SWA + Partial RoPE + LN Scale + XSA4 (val_bpb: 1.1270)

**val_bpb: 1.1270** (3-seed mean, sliding window stride=64) | **15.5 MB** | 8xH100 SXM, 600s

## Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared activation
- Exclusive Self Attention (XSA) on last 4 layers
- Partial RoPE (16/64 dims) with NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate + BigramHash (2048 buckets, dim=128)
- FlashAttention 3 (Hopper)
- Orthogonal init with proj scaling by 1/sqrt(2*num_layers)
- Logit softcap 30.0, tied embeddings

## Training

- Muon optimizer (matrices): lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3000 iters (wallclock-based)
- **Tight SWA**: every 50 steps when scale<0.2 (~12 checkpoints from last ~600 steps)

## Quantization

- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | Sliding BPB (s64) | Artifact |
|------|-------|-------------------|----------|
| 1337 | 7,094 | 1.1268 | 15,498,865 |
| **42** | **7,099** | **1.1265** | **15,469,409** |
| 7 | 7,099 | 1.1277 | 15,482,613 |

**Mean: 1.1270 | Std: 0.0006**

## Key Innovation: Tight SWA

Standard SWA (scale<0.5) averages stale checkpoints that hurt final quality. Tight SWA restricts checkpoint collection to scale<0.2 (last ~600 steps), every 50 steps. This eliminates the SWA quality penalty while maintaining quantization-friendly weight averaging. Post-SWA BPB equals pre-SWA BPB (zero penalty).

## Run Command

```bash
pip install "git+https://github.com/Dao-AILab/flash-attention.git#egg=flash-attn-3&subdirectory=hopper" --no-build-isolation
pip install zstandard

NUM_LAYERS=11 MLP_MULT=3.0 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
SWA_ENABLED=1 SWA_EVERY=50 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 ADAM_WD=0.04 MUON_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
