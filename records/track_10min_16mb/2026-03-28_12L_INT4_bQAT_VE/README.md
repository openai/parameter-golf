# 12L INT4 bQAT + EMA Fix + Value Embeddings

**val_bpb: 1.1588** (seed 1, full TTT) | **16.29 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM)

| Seed | step_avg | steps | Pre-quant val/bpb | Post-quant bpb | Post-TTT bpb | Artifact |
|------|----------|-------|-------------------|----------------|--------------|----------|
| 1    | ~137ms   | ~4380 | 1.1754            | 1.1643         | **1.1588**   | 16,290,425 |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 12 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 10240 buckets, INT4 bQAT |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| Skip | U-Net skip connections |
| resid_mix | Learnable x/x₀ blend |
| Weight avg | EMA(0.997) with QAT-activation reset |
| Quantization | INT4 MLP + INT4 bigram + INT6 attn + zstd |
| QAT trigger | Wallclock fraction (65% of budget) |
| Value Embeddings | ve_dim=128, layers 10-11 |
| TTT | Legal score-first, lr=0.002, 3 epochs |

## Key Addition: Value Embeddings

Value embeddings reinject token identity into the V vectors at specific attention layers. At layers 10 and 11, a shared embedding `ve_shared` (vocab×128) is looked up per token and projected to kv_dim (256), then added to the raw V output before attention:

```
v_raw = c_v(x)
v_embed = ve_shared.proj(ve_shared.embed(tokens))  # (batch, seq, kv_dim)
v = v_raw + v_embed
```

The shared embedding allows all VE layers to benefit from a single (vocab, 128) table without per-layer weight cost.

**Effect:** VE improved quality by ~0.014 BPB per step vs baseline at step 2000 (1.2344 vs 1.2481). Despite 640 fewer total steps (4380 vs 5021 for v4_h100 seed 1), the per-step quality gain resulted in a new best ttt_bpb.

## Comparison with Previous Best

| Run | steps | Pre-quant | Post-quant | TTT bpb |
|-----|-------|-----------|------------|---------|
| v4_h100 seed 1 | 5021 | 1.1683 | 1.1703 | ~1.165 |
| v4_h100 seed 3 | — | 1.1729 | 1.2002 | 1.1594 |
| **v7_ve seed 1** | ~4380 | 1.1754 | **1.1643** | **1.1588** |

Note: v7_ve's post_quant (1.1643) is better than its pre-quant checkpoint (1.1754) because the model continued improving during QAT after the last val checkpoint.

## Size Budget

| Component | Bytes |
|-----------|-------|
| Base model (int4/int6/zstd) | ~15,967,640 |
| Value embeddings (int8/int4/zstd) | ~322,785 |
| **Total** | **16,290,425** |

Budget: 16,777,216 bytes (16MB) — **486,791 bytes (475KB) margin**

## Run Command

```bash
SEED=1 VALUE_EMBED_LAYERS=2 VALUE_EMBED_DIM=128 \
LATE_QAT_FRAC=0.65 VAL_LOSS_EVERY=1000 \
NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- LeakyReLU² activation: PR #493, PR #518
- XSA (Cross-layer Shared Attention): PR #414
- EMA weight averaging: PR #374
- Legal TTT recipe: PR #461
- INT5/INT6 QAT with STE: PR #317, PR #374
- BigramHash embedding: PR #320
- U-Net skip connections: PR #363
- resid_mix: prior work in this repo
- Value embeddings: PR #549
