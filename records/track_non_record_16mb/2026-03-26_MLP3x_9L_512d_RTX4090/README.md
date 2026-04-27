# Non-Record Submission: MLP3x 9L 512d (RTX 4090)

**Best score from automated hyperparameter sweep on a single RTX 4090.**

## Results

| Metric | Value |
|--------|-------|
| val_bpb (int8+zlib) | **1.4530** |
| val_bpb (pre-quant) | 1.4518 |
| val_loss | 2.4533 |
| Steps | 960/1000 (wallclock cap at 600s) |
| Parameters | 21,778,504 |
| Submission size | 15.35 MB (int8+zlib) |
| GPU | 1x RTX 4090 24GB |

## Architecture

Standard GPT with:
- **9 layers**, 512 model dim, 8 attention heads, 4 KV heads (GQA)
- **3x MLP multiplier** (1536 FFN hidden dim instead of default 1024)
- SentencePiece tokenizer (1024 vocab, FineWeb 10B)
- Muon optimizer + int8 quantization + zlib compression

## Key Insight

In our automated sweep across 11 configurations, **MLP3x consistently outperformed deeper models** (12L) and other variants (QK gain, warmup schedules, narrow/wide). At this parameter budget (~22M), investing in wider feedforward layers beats adding more attention layers.

## Sweep Leaderboard (all 1x RTX 4090)

| Config | val_bpb | Notes |
|--------|---------|-------|
| **mlp3_9L_512d** | **1.4530** | this submission |
| deep_12L_512d | 1.4613 | More layers, same params |
| base_9L_512d | 1.4659 | Default MLP2x |
| warm50_9L_512d | 1.4661 | 50-step warmup |
| qk2_9L_512d | 1.4672 | QK gain=2.0 |
| shallow_6L_512d | 1.5071 | Too few layers |

## Why Non-Record

Trained on a single RTX 4090 rather than 8xH100s. The architecture and training recipe are standard — our contribution is the systematic sweep showing MLP width > depth at this scale, and demonstrating competitive results on consumer hardware.

## Reproduction

```bash
env NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
    ITERATIONS=1000 MAX_WALLCLOCK_SECONDS=600 \
    python3 train_gpt.py
```
