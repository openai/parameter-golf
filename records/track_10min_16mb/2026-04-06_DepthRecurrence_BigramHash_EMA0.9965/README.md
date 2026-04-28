# 11L Depth Recurrence + BigramHash + EMA 0.9965 — val_bpb 1.0980

**val_bpb: 1.09796317** (3-seed mean, std 0.0008) | **~14.6 MB** | 8xH100 SXM, 600s

## Summary

Depth recurrence architecture from PR #1334/#1421 combined with BigramHash(1536, dim 112) from the cumulative competition stack. 11 physical transformer layers with layers 4,5 repeating once, yielding 13 virtual layers. Recurrence activates at step 3000. EMA decay 0.9965.

BigramHash added ~0.001 BPB improvement over the base recurrence architecture: 1.0999 → 1.0989 (seed 1337), at a cost of ~230K additional parameters and 270KB artifact size.

SP1024 tokenizer used. SP4096 was unavailable in the public data manifest at time of submission.

## 3-Seed Results (8xH100 SXM)

| Seed | Steps | Pre-quant BPB | Sliding BPB (s64) | val_loss (nats) | Artifact |
|------|-------|---------------|-------------------|-----------------|----------|
| 1337 | 5347 | 1.1104 | **1.09885595** | 1.85537224 | 14,597,964 B |
| 42 | 5545 | 1.1089 | **1.09733806** | 1.85280934 | 14,564,857 B |
| 2024 | 5554 | 1.1097 | **1.09769550** | 1.85341287 | 14,561,630 B |
| **Mean** | | **1.1097** | **1.09796317 (std 0.00079)** | **1.85386482** | |

## Statistical Significance

Current merged SOTA: **1.11473509 BPB** / **1.88217853 nats** (PR #1019, 3-seed mean).

| Metric | Value |
|--------|-------|
| Delta (nats) | **-0.02831371** |
| Delta (BPB) | **-0.01677350** |
| Welch's t-statistic | **-30.75** |
| Welch-Satterthwaite df | **3.42** |
| p-value | **<< 0.001** |
| Exceeds 0.005 nats threshold | Yes (5.7x) |

## Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- **Depth recurrence**: layers 4,5 repeat (virtual 13 layers), activated at step 3000
- **BigramHash**: 1536 buckets, dim 112, with SmearGate
- Skip gates (learnable residual gating on U-Net skip connections)
- Parallel residuals (layers 7+, attention and MLP in parallel lanes)
- Value Embedding (dim=128, layers 9,10)
- SP1024 tokenizer (SentencePiece BPE)
- MuonEq-R (row normalization before NS5)
- QK-Gain (learnable per-head Q scaling, init=5.0)
- Tied embeddings, logit softcap=30.0, partial RoPE (16/64 dims)
- XSA on all 11 layers

## Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer: lr=0.02, momentum=0.99, WD=0.09, backend_steps=5
- Adam (head): lr=0.008 | AdamW (embeddings): lr=0.6, WD=0.09
- AdamW (scalars): lr=0.02, WD=0.02
- Gradient clip: 0.3, Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 66.7% of training
- **EMA**: decay=0.9965
- Wallclock cap: 600s (590s effective, 10s reserved for GPTQ)

## Reproduction

```bash
# Setup
pip install --break-system-packages zstandard brotli
python3 data/cached_challenge_fineweb.py --variant sp1024

# Training (8xH100 SXM, ~600s)
SEED=1337 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-06_DepthRecurrence_BigramHash_EMA0.9965/train_gpt.py
```

All hyperparameters use script defaults. Key non-default env vars are not required — the script is self-contained. Replace `SEED=1337` with `SEED=42` or `SEED=2024` to reproduce the other seeds.

## Quantization

- GPTQ int6, percdamp=0.05, 64 calibration batches
- Calibration data: AR self-generated from the trained model (not validation data)
- No selective pruning needed (artifact fits under 16 MB)
- Brotli compression

## Comparison: BigramHash vs Vanilla

| Variant | Sliding BPB (s1337) | Artifact | Model Params |
|---------|-------------------|----------|-------------|
| Vanilla (PR #1421 base) | 1.0999 | 14,327,531 | 32,435,292 |
| **+ BigramHash** | **1.0989** | 14,597,964 | 32,665,181 |

BigramHash provides a small but consistent improvement (+0.001 BPB) at minimal artifact cost.

## Attribution

- Base architecture + depth recurrence: PR #1334 by @aryanbhosale
- EMA tuning (0.9965): PR #1421 by @X-Abhishek-X
- BigramHash + SmearGate: from the cumulative competition stack (PRs #609, #1019)
