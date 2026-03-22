# pcloadloveletter v5

**v5 = v4 + new meta techniques + novel compression pipeline**

## New in v5 (vs v4)

### Standard meta techniques (what top 10 all use now):
- **Partial RoPE** (16/64 dims) — only 25% of head dims get positional encoding
- **LN Scale** (1/sqrt(layer_idx+1)) — progressive layer norm damping for deeper layers
- **XSA on last 4 layers** — exclusive self-attention removes self-value bias
- **Late QAT** (STE int6 at lr_scale < 0.1) — quantization-aware training in final 10%
- **Tight SWA** (scale < 0.2) — only fresh checkpoints, zero averaging quality penalty
- **LR bump** 0.02 → 0.025

### Novel compression (our differentiation — nobody else does this):
- **Per-tensor k-means codebook quantization** — non-uniform quantization levels matched to actual weight distribution
- **Mixed codebook sizes** — CB-48 for MLP, CB-80 for attention QKV, CB-64 for attention proj
- **Huffman entropy coding** — distribution-aware encoding beats zstd by 1.66 MB
- **Custom binary format** (PCLL) — no pickle, no ZIP, minimal overhead
- **Estimated savings: 3.82 MB (21%) vs baseline int6+zstd**

## Run Commands

```bash
# 1x3080 local test (60s)
MAX_WALLCLOCK_SECONDS=60 TRAIN_BATCH_TOKENS=32768 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 official run (10 min)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Disable novel compression (fallback to int6+zstd)
USE_NOVEL_COMPRESSION=0 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Environment Variables (new in v5)

| Variable | Default | Description |
|----------|---------|-------------|
| `ROPE_DIMS` | 16 | Number of head dims to apply RoPE to (partial RoPE) |
| `LN_SCALE` | 1 | Enable LN Scale (1/sqrt(layer_idx+1)) |
| `XSA_LAYERS` | 4 | Number of final layers to apply XSA |
| `LATE_QAT` | 1 | Enable Late QAT (STE int6 fake-quantization) |
| `LATE_QAT_THRESHOLD` | 0.1 | LR scale threshold to enable QAT |
| `USE_NOVEL_COMPRESSION` | 1 | Use novel codebook+Huffman pipeline |
| `CODEBOOK_MLP` | 48 | Codebook levels for MLP tensors |
| `CODEBOOK_ATTN_QKV` | 80 | Codebook levels for attention QKV |
| `CODEBOOK_ATTN_PROJ` | 64 | Codebook levels for attention output proj |
