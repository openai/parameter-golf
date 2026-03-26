# Non-record: 6-Technique Stack — Catalytic Residuals + Value Residual + Gated Attention + BigramHash(10240) + 12L (val_bpb=1.1690)

First submission to combine 6 independently-proven architecture improvements that have never been stacked together in a single entry.

## Results (8xH100 SXM)

| Metric | Value |
|--------|-------|
| **Sliding window val_bpb** | **1.1690** |
| Post-quant roundtrip val_bpb | 1.2043 |
| Pre-quant val_bpb | 1.1911 |
| Steps | 6,981 |
| Step avg | 85.78 ms |
| Training time | 598.8s |
| Artifact size | 15,372,888 bytes (15.3 MB) |
| Compressed model | 15,312,258 bytes (int6+zstd) |

## Architecture

- **12 layers**, 512 dim, 8 heads / 4 KV heads (GQA), 3x MLP (ReLU-squared)
- Vocab 1024 (SentencePiece BPE), seq len 1024, tied embeddings

## Novel Technique Combination

Each technique was proven in a separate PR with controlled ablation data. This is the first submission to combine all 6:

| Technique | Source | Measured Impact | Description |
|-----------|--------|-----------------|-------------|
| **Catalytic Residuals** | PR #450 | -0.024 bpb | `x + c * f(x)` with learned per-dim scalar c (init 1.0). Zero compute overhead. |
| **Value Residual (ResFormer)** | PR #413, arXiv:2410.17897 | -0.015 bpb | Cache layer-0 V vectors, mix into subsequent layers via learned scalars. |
| **Gated Attention** | PR #413, arXiv:2505.06708 | -0.003 bpb | Per-head sigmoid gate after attention output. |
| **BigramHash(10240)** | PR #450 | -0.070 bpb (vs 2048) | Hash-based bigram embedding with 10240 buckets. |
| **12 Layers** | PR #450 | -0.023 bpb (vs 11L) | Deeper model within 16MB budget. |
| **3x MLP** | Merged SOTA | Standard | 3x expansion vs baseline 2x. |

## Additional Techniques

- **OrthoInit**: Orthogonal weight init with muP-style projection scaling
- **Muon WD=0.04**: Decoupled weight decay on both Muon and AdamW
- **SWA**: Stochastic weight averaging from last 20% of warmdown
- **Late QAT (threshold 0.25)**: STE int6 fake-quantize during warmdown
- **Sliding window eval (stride 64)**: Overlapping windows for final BPB
- **Logit softcap 30.0**
- **Int6+zstd compression**: Mixed int6 (mlp+attn) / int8 (embed) with zstandard level 22

## Run Command

```bash
pip install sentencepiece zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters set as defaults. No env vars needed for the standard run.

## Dependencies

- PyTorch >= 2.5 (native GQA via `enable_gqa=True` in SDPA)
- sentencepiece
- zstandard
- numpy
