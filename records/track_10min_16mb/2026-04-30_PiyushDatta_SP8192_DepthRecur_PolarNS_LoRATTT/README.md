# SP8192 + Depth Recurrence + Polar Express NS + Phased LoRA TTT

## Summary

11-layer GPT with SP8192 tokenizer, MLP 4x, depth recurrence (layers 3-5 looped once), parallel residuals, Polar Express Newton-Schulz optimizer, and phased LoRA test-time training.

**val_bpb: 1.09085** (3-seed mean, 8xH100, quantized sliding window)

## Key Techniques

- **SP8192 tokenizer**: 8x larger vocabulary vs SP1024 baseline
- **Depth recurrence**: Layers 3-5 run twice (14 effective passes from 11 unique layers), activated at 45% of training
- **Polar Express NS**: Per-iteration minimax-optimal Newton-Schulz coefficients for Muon optimizer
- **Parallel residuals**: Attention and MLP computed from same input in layers 7+
- **MuonEq-R optimizer**: Row-normalized Muon with momentum 0.95
- **SDClip GPTQ**: Hessian-weighted clip ranges for int6 quantization + int8 embeddings
- **SWA**: Stochastic Weight Averaging (every step during warmdown)
- **Half-batch training**: 393K tokens/batch for more gradient steps
- **Brotli compression**: Better compression ratio than LZMA for model weights
- **Phased LoRA TTT**: Score-first test-time training with batched LoRA adaptation

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 11 (14 effective with depth recurrence) |
| Dimension | 512 |
| Heads | 8 (4 KV heads, GQA) |
| MLP multiplier | 4.0x |
| Activation | LeakyReLU(0.5)^2 |
| Vocab size | 8192 (SentencePiece) |
| Quantization | int6 (weights) + int8 (embeddings) |
| Compression | Brotli |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (matrix) + AdamW (scalars, embeddings) |
| Matrix LR | 0.028 |
| Muon WD | 0.095 |
| Embed WD | 0.085 |
| Warmdown | 72% of training |
| SWA | Every step, start at scale < 0.12 |
| MIN_LR | 0.10 |
| Batch tokens | 393,216 |
| Max wallclock | 600s |

## Reproduction

```bash
# On 8xH100:
cd /workspace/parameter-golf
bash records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT/run_final_submission.sh --nproc 8

# On 4xA100 (local testing, TTT will be slow):
bash records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT/run_final_submission.sh --nproc 4
```

## Attribution

- SP8192 + GPTQ embeddings + SDClip: @clarkkev (PR #1394)
- Depth recurrence: @dexhunter (PR #1331, #1437)
- Parallel residuals: @Robby955 (PR #1412), @msisovic (PR #1204)
- Legal TTT framework: @abaybektursun (PR #549), @dexhunter (PR #1413)
- Polar Express NS: custom implementation (arxiv 2505.16932)
- Phased LoRA TTT: @dexhunter (PR #1626), @romeerp (PR #1610)
