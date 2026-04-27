# Pre-Enrichment + EMA-GPU + SmearGate + XSA4

val_bpb: **1.1478** (sliding window, stride=64) | 14.94 MB | 8xH100 SXM, 600s

### 3-Seed Results

| Seed | Steps | val_bpb (sliding) | Artifact |
|---|---|---|---|
| 1337 | 9,268 | 1.1478 | 14,942,971 |
| 42 | 9,318 | 1.1468 | 14,922,769 |
| 3011 | 9,322 | 1.1463 | 14,939,305 |
| **Mean** | — | **1.1470** | — |
| **Std** | — | **0.0008** | — |

## Architecture

- **Model**: 10L, 512d, 8H/4KV GQA, MLP 3x, tied embeddings
- **GELU Pre-Enrichment** (512->768->512): wider nonlinear transformation before transformer blocks
- **XSA** on last 4 layers: removes self-value bias (arXiv:2603.09078)
- **SmearGate**: per-dim gate blending each token with previous token
- **BigramHash** (2048x128): hash-table embedding for token bigrams
- **U-Net skip connections**: encoder-decoder with learned skip weights
- **EMA** (decay=0.997) on GPU: 37% faster training (64.7ms vs 101ms/step)
- **Int6 QAT + lzma**: 14.94 MB artifact, quant gap 0.004
- **Sliding window eval**: stride=64, seq_len=2048

## Training

Muon+AdamW, WD=0.04, matrix_lr=0.025, warmdown=3500, batch=524K, seq=2048.
9,268 steps in 600s at 64.7ms/step.

## Key Metrics

| Metric | Value |
|---|---|
| val_bpb (sliding window) | 1.1478 |
| Post-quant val_bpb (standard) | 1.1690 |
| Pre-quant val_bpb | 1.1646 |
| Quant gap | 0.004 |
| Training time | 600,031ms (9,268 steps at 64.7ms) |
| Artifact size | 14,942,971 bytes |
| Model parameters | 25,254,992 |

## Credits

- Muon optimizer — modded-nanogpt baseline (kellerjordan)
- SmearGate + BigramHash — PR #65 (@aquariouseworkman)
- XSA — arXiv:2603.09078; GQA-aware PR #265 (@unnir)
- EMA + GPTQ-lite + warmdown tuning — PR #414 (@signalrush)
- Overtone init — modded-nanogpt baseline
- GELU Pre-Enrichment — original to this submission
- EMA on GPU — original to this submission

## Reproduction

```
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
8xH100 SXM, 600s training + ~120s eval.
