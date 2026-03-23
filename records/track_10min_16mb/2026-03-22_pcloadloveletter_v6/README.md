# pcloadloveletter v6 — Artie AI

**val_bpb: 1.0487** (8xH100 SXM, seed 1337)

## Architecture

11L transformer, d=512, 8 query heads / 4 KV heads (GQA), MLP hidden=1500, tied embeddings, vocab 1024.

## What's Different

### Novel Compression Pipeline (our key contribution)

Everyone in the competition uses the same compression: uniform int6 quantization in int8 containers + zstd. We built something better:

1. **Per-tensor k-means codebook quantization** — non-uniform quantization levels matched to actual weight distributions via k-means clustering. Different codebook sizes per tensor type (CB-48 for MLP, CB-80 for attention QKV, CB-64 for projections) based on measured sensitivity.

2. **Huffman entropy coding** of codebook indices — exploits the non-uniform index distribution that general-purpose compressors (zstd) miss. Huffman beats zstd by 1.66 MB on weight data because it's distribution-aware.

3. **Custom binary format (PCLL)** — compact serialization with per-tensor metadata, followed by zstd-22 final compression.

**Result: 14.12 MB artifact** (vs 18+ MB with standard int6+zstd on the same model). Saves 21% — enough headroom to fit architectural additions that wouldn't otherwise fit under 16 MB.

Prior work (PR #212) tested codebook + zstd and got 25% *larger* artifacts — the Huffman stage is what makes codebook compression viable.

### Training Techniques

- **EMA** (decay=0.997) replacing SWA
- **Value Residual** (arXiv:2410.17897) — cache layer-0 V, blend via learned lambda. 22 params, -0.015 BPB.
- **Gated Attention** (arXiv:2505.06708) — per-head sigmoid gate after SDPA. -0.003 BPB.
- **LeakyReLU(0.5)^2** activation (from PR #518)
- **Partial RoPE** (16/64 dims), **LN Scale**, **XSA** on last 4 layers
- NorMuon optimizer (MATRIX_LR=0.03, WD=0.04)

### Test-Time Training

AdamW TTT with cosine lr schedule and per-layer lr groups:
- 10 epochs, lr=0.001, cosine decay
- Output projections (c_proj, mlp.proj): 3x lr
- MLP FC: 0.5x lr
- All params unfrozen, grad clip 1.0

## Validated Results

| Metric | Value |
|--------|-------|
| Training steps | 5,364 (112ms/step) |
| Pre-quant val_bpb | 1.1511 |
| Post-quant (codebook) | ~1.16 |
| **Post-TTT val_bpb** | **1.0487** |
| Artifact size | 14.12 MB (88.3% of cap) |
| Compression savings | 3.9 MB vs int6+zstd (21%) |

## Reproduction

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install sentencepiece zstandard huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-22_pcloadloveletter_v6/train_gpt.py
```

## Team

Built by [Artie AI](https://github.com/NotADevIAmaMeatPopsicle)
