# 11L PartialRoPE + LNScale + EMA + SWA + TTT (non-record)

Non-record submission for the Parameter Golf challenge. This run was tested on **1×H100 PCIe for ~107 minutes** (approximately equivalent to 8×H100 SXM for 10 minutes).

## Architecture

- **11 transformer layers**, d=512, 8 heads / 4 KV heads, 3× ReluSquared MLP
- **U-Net skip connections**: encoder-decoder style with learnable skip weights
- **Partial RoPE**: rotary on 16 of 64 head dims for position-free generalization
- **LN Scale**: RMSNorm damped by 1/sqrt(layer+1) for deep gradient stability
- **SmearGate**: per-dim gate blending current + previous token embeddings
- **BigramHash(2048, dim=64→512)**: hash-based bigram context embeddings
- Tied input/output embeddings

## Training

- Muon optimizer (Newton-Schulz) for 2D weights, momentum warmup 0.85→0.99
- Adam (beta1=0.9, beta2=0.95) for scalars/embeddings, WD=0.04
- Wallclock-aware cosine warmdown over last ~3000 steps
- Orthogonal init with muP output-projection scaling
- EMA (decay=0.997) + SWA (last 40% of training)

## Compression

- Uniform int5 per-row quantization (both MLP and attn) + int8 fallback
- zstd-22 compression
- **Artifact size: 15.4MB ✅ (under 16MB limit)**

## Evaluation

- Sliding window with stride=64 for near-max context scoring
- Full-model SGD TTT: 3 epochs over val, first 2 blocks frozen

## Key Metrics

| Metric | Value |
|---|---|
| val_loss (pre-TTT) | 2.0611 |
| val_bpb (pre-TTT) | 1.2207 |
| Training steps | 3374 |
| Training time | 6400210ms |
| SWA count | 1197 |
| Model params | 26,666,073 |
| Artifact bytes | 16,132,620 |
| Code bytes | 49,461 |
| Total bytes | 16,182,081 |

## Included Files

- `train_gpt.py` — training script
- `train.log` — training log
- `submission.json` — submission metadata
- `README.md` — this file
