# Non-Record Submission: Autoresearch 8L 512d (RTX 4090)

**Best score from autoresearch framework on a single RTX 4090.**

## Results

| Metric | Value |
|--------|-------|
| val_bpb | **1.3718** |
| Steps | 107 (wallclock cap at 300s) |
| Parameters | 50,332,176 |
| Peak VRAM | 11,843 MB |
| Training time | 301.9s |
| Total time | 386.4s |
| MFU | 4.07% |
| GPU | 1x RTX 4090 24GB |

## Architecture

Autoresearch pretraining framework (cherry-picked from nanochat):
- **8 layers**, 512 model dim, 4 attention heads, 4 KV heads
- **8192 BPE vocab** (SentencePiece)
- **Sequence length 2048** with sliding-window attention (SSSL pattern)
- **Value embeddings** (16.7M params — large learned value table)
- Muon + AdamW hybrid optimizer
- Gradient accumulation 16 steps, batch size 16 per device

## Key Insight

The autoresearch framework from nanochat uses a **value_embeds** table (16.7M params) alongside standard token embeddings (4.2M) — allocating ~33% of total parameters to learned value representations. Combined with the SSSL sliding-window attention pattern and 8192 BPE tokenizer (vs 1024 in our previous submission), this achieves **1.3718 bpb** — a 5.6% improvement over our previous best of 1.4530 bpb.

## Training Curve

Loss decreased smoothly from 9.01 → 3.92 over 107 steps at ~167K tok/s.

## Previous Submission

This supersedes our MLP3x 9L 512d submission (1.4530 bpb, PR #854).

## Reproduction

```bash
cd autoresearch
uv run train.py  # defaults: 8L 512d 4h 4kv 8192vocab batch16 300s budget
```
