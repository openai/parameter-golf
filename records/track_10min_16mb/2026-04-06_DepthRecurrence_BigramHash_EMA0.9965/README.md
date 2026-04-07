# 11L Depth Recurrence + BigramHash + EMA 0.9965 — val_bpb 1.0980

**val_bpb: 1.0980** (3-seed mean, std 0.0008) | **~14.6 MB** | 8xH100 SXM, 600s

## Summary

Depth recurrence architecture from PR #1334/#1421 combined with BigramHash(1536, dim 112) from our prior SOTA stack. 11 physical transformer layers with layers 4,5 repeating once, yielding 13 virtual layers. Recurrence activates at step 3000. EMA decay 0.9965.

BigramHash added ~0.001 BPB improvement over the base recurrence architecture (1.0989 -> 1.0980 mean), at a cost of ~230K additional parameters and 270KB artifact size.

SP4096 tokenizer was unavailable in the public data manifest, so SP1024 was used instead. With SP4096, results would likely be ~0.01 BPB better.

## 3-Seed Results (8xH100 SXM)

| Seed | Pre-quant BPB | Sliding BPB (s64) | Artifact |
|------|---------------|-------------------|----------|
| 1337 | 1.1104 | **1.0989** | 14,597,964 B |
| 42 | 1.1089 | **1.0973** | 14,564,857 B |
| 2024 | 1.1097 | **1.0977** | 14,561,630 B |
| **Mean** | **1.1097** | **1.0980 (std 0.0008)** | |

Current merged SOTA: **1.1147** (PR #1019). Delta: **-0.0167 BPB**.

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

## Quantization

- GPTQ int6, percdamp=0.05, 64 calibration batches
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
