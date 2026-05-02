# 11L LeakyReLU² + Early QAT@0.5 + GPTQ-lite + EMA

**val_bpb: 1.1387** (sliding window stride=64, 3-seed mean) | **~15.6 MB** | 8xH100 SXM, 600s

## Key Innovation: Early QAT

The main finding is that starting STE fake-quantization much earlier during warmdown (**LR scale < 0.5**, yielding ~1400 QAT steps) dramatically reduces the post-quantization gap compared to the typical threshold of 0.15 (which often yields only a handful of QAT steps before wallclock cutoff).

| QAT Threshold | QAT Steps | Quant Gap (BPB) |
|---------------|-----------|-----------------|
| 0.15 (typical) | ~1 | 0.28 |
| **0.5 (ours)** | **~1400** | **0.004** |

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 5621 | 1.9237 | **1.1393** | 15.53 MB |
| 42 | 5599 | 1.9225 | 1.1386 | 15.61 MB |
| 2025 | ~5600 | 1.9219 | 1.1383 | 15.52 MB |

**Mean: 1.1387 | Std: 0.0005**

## Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion (1536 hidden), LeakyReLU(0.5)² activation
- U-Net skip connections (5 encoder, 6 decoder)
- Tied embeddings, logit softcap=30
- RoPE, RMSNorm
- Flash attention (`is_causal=True`) on all layers

## Training

- Muon optimizer (matrices): lr=0.025, momentum warmup 0.85→0.95, WD=0.04
- AdamW (embeddings/scalars): lr=0.035/0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations (wallclock-based)
- EMA: decay=0.997, every step, applied before export
- **Early QAT**: int6 STE fake-quantization when LR scale < 0.5

## Quantization & Export

- GPTQ-lite: per-row optimal clip percentile search (5 candidates) for int6
- Int6 per-row for attention + MLP weights
- Int8 per-row for embeddings
- Control tensors in fp32 passthrough
- zstd level 22 compression (lzma-6 fallback)

## Acknowledgments & Prior Work

This submission builds on techniques introduced by prior Parameter Golf entries:

| Technique | First introduced by | PR/Record |
|-----------|-------------------|-----------|
| LeakyReLU(0.5)² activation | @abaybektursun | [PR #549](records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/) |
| GPTQ-lite clip search | @signalrush | [PR #414](records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/) |
| EMA weight averaging | @jfprincz | [PR #198](records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/) |
| Int6 STE QAT | @aruniyer | [PR #198](records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/) |
| Sliding window eval (stride=64) | @mattqlf | [PR #50](records/track_10min_16mb/2026-03-19_SlidingWindowEval/) |
| 3x MLP expansion | @aquariouseworkman | [PR #198](records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/) |
| Muon + AdamW weight decay | @notapplica | [PR #148](records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/) |
| Warmdown 3500 | @signalrush | [PR #414](records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/) |
| Muon optimizer | [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Baseline |

**Our novel contribution**: Early QAT at LR scale 0.5 (vs typical 0.15), yielding ~1400 steps of quantization-aware training and reducing the post-quant gap from 0.28 to 0.004 BPB.

## Run Command

```bash
SEED=1337 RUN_ID=submission \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
