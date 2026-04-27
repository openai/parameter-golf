# [10min/16mb] David Ghazaryan — MoE + BigramHash4096

**val_bpb: 1.1180** (3-seed mean) | 8×H100 SXM | 600s

## Results

| Seed | val_bpb | Artifact (bytes) |
|------|---------|-----------------|
| 1337 | 1.11764880 | 15,873,596 |
| 42   | 1.11891002 | 15,893,104 |
| 2025 | 1.11742168 | 15,908,116 |
| **mean** | **1.11799350** | 15,891,605 |

## Novel Contributions

### 1. BigramHash 4096
Expanded bigram hash table from SOTA's 3072 to 4096 buckets.
Provides richer local context signal at the embedding stage.

### 2. Mixture-of-Experts MLP (first in this repo)
Replaces standard MLP with 4 experts + top-2 routing.
Same active parameters but adds expert specialisation.

## Architecture

| Component | Setting | Introduced by |
|-----------|---------|---------------|
| Layers | 11 (512d, 8 heads, 4 KV) | Baseline |
| MLP | 3x LeakyReLU(0.5)² | PR #493 |
| XSA | All 11 layers | PR #478 |
| EMA | decay=0.997 | PR #374 |
| Partial RoPE | 16/64 dims | PR #315 |
| LN Scale | 1/√(layer+1) | PR #315 |
| GPTQ | Full Hessian AR self-gen | PR #1019 |
| BigramHash | 4096 buckets dim=96 | This PR |
| MoE MLP | 4 experts top-2 | This PR |
| Compression | int6 + lzma | PR #414 |

## Requirements

```bash
pip install sentencepiece zstandard
pip install flash_attn_3 --find-links \
  https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

## Run Command

```bash
BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=96 WARMDOWN_ITERS=4000 TARGET_MB=15.9 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware
8× H100 80GB HBM3 (YSU HPC Cluster, YerevaNN/Eleveight AI Program)
