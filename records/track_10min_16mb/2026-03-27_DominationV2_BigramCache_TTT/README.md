# DominationV2 + BOS-Reset Bigram Cache + TTT

**val_bpb: 1.1382** (3-seed mean, std 0.0010) | **~15.5 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | val_bpb | Artifact |
|------|----------|-------|---------|----------|
| 1337 | 69.7ms | 8,611 | **1.1371** | 15,504,722 |
| 42 | 69.8ms | 8,605 | **1.1385** | 15,579,418 |
| 2025 | 69.7ms | 8,621 | **1.1389** | 15,505,762 |
| **Mean** | **69.7ms** | **8,612** | **1.1382** | |

### Timing Budget

| Phase | Time |
|-------|------|
| Training (8,611 steps @ 69.7ms) | 600s |
| TTT (3 epochs) | ~10s |
| Sliding window + cache eval | ~223s |
| **Total eval** | **~233s** |

## BOS-Reset Bigram Cache

An eval-time bigram cache applied during sliding window evaluation, after quantization roundtrip and TTT.

For each scored token, the cache tracks bigram counts from already-scored tokens within the current document and blends with model probabilities:

```
p_final = (1 - alpha_eff) * p_model + alpha_eff * p_cache

p_cache  = count(prev, target) / count(prev)
alpha_eff = 0.20 * count / (count + 8)        scales with observed data
alpha_eff *= (entropy / max_entropy)           higher when model is uncertain
```

Cache resets at every BOS token (document boundary). Updated only after each token is scored (score-first, same ordering as TTT in PR #549).

## Architecture

DominationV2 stack:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x relu² |
| U-Net | 5 encoder + 6 decoder with skip connections |
| XSA | Last 4 layers |
| SmearGate | Per-dimension blend with previous token |
| BigramHash | 2048 buckets, dim=128 |
| OrthoInit | Orthogonal init with depth scaling |
| EMA | Decay=0.997 |
| Quantization | Mixed int6/int8 + zstd-22 |
| TTT | 3 epochs, lr=1e-4 |

### Cache Settings

| Parameter | Value |
|-----------|-------|
| CACHE_ALPHA | 0.20 |
| CACHE_TAU | 8.0 |
| CACHE_ENTROPY_POWER | 1.0 |
| Eval stride | 64 |

## Run Command

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
pip install zstandard

cd records/track_10min_16mb/2026-03-27_DominationV2_BigramCache_TTT

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- DominationV2 base: built on upstream PR #64 and PR #198
- Bigram cache: inspired by classical cache language models (Grave et al., 2016)
- TTT: adapted from PR #461
