# Non-record: Wider Model (d=576) + Int5 QAT with Early Activation

**val_bpb: 1.1418** (sliding window, stride=64) | **15.7 MB** artifact | 8xH100 SXM, 600s

## Approach

Train a **wider 27M-param model** (d=576 vs standard d=512, +23% parameters) and compress to int5 (32 levels) instead of int6 (64 levels). The wider model converges to lower pre-quant loss, and early int5 QAT activation (threshold 0.50 vs standard 0.10) gives the model ~1,700 steps to adapt to the coarser quantization grid.

### The train-large-compress principle

Everyone trains ~22M params sized to fit 16MB at int6. We demonstrate that training a larger model and compressing it more aggressively can match or beat smaller models at the same artifact size. Li et al. (ICML 2020) showed compressed large models beat lightly compressed small models — this submission validates that principle for the parameter golf setting.

| | Standard approach | This submission |
|---|---|---|
| Model dim | 512 | **576** |
| Parameters | 22M | **27M** (+23%) |
| Heads / KV | 8 / 4 | **9 / 3** |
| Quantization | int6 (64 levels) | **int5 (32 levels)** |
| QAT activation | Last ~4% | **Last ~25%** |

## Architecture

- 11 layers, d=576, 9 attention heads (head_dim=64), 3 KV heads (GQA 3:1)
- MLP 3x expansion (hidden=1728), relu-squared activation
- SmearGate + BigramHash(4096, dim=128)
- XSA on last 4 layers
- Partial RoPE (16/64 dims)
- U-Net skip connections
- OrthoInit with muP-scaled output projections

## Training

- Muon optimizer (WD=0.04, momentum=0.99)
- SWA weight averaging
- FA3 for attention throughput
- Warmdown: 3500 iters
- Batch: 786K tokens, seq_len=2048
- Int5 STE fake-quantization activated when lr_scale < 0.50
- ~7,000 steps in 600s on 8xH100 SXM

## Post-training compression

- Int5 per-row quantization for MLP and attention weights
- Int8 for embeddings, fp16 for tied embedding
- zstd-22 compression
- Artifact: 15.7 MB

## Results

| Seed | val_bpb (sliding s64) | artifact_bytes |
|------|----------------------|----------------|
| 1337 | **1.1418** | 15,713,507 |

Pre-quant val_bpb: 1.1515. Quantization gap: 0.010.

Note: single-seed submission. Artifact size varies by seed (15.2-16.5 MB range). Additional seeds exceeded the 16 MB limit.

## What's novel

1. **Wider model at lower bit-width**: 27M params at int5 instead of 22M at int6. The extra capacity outweighs the coarser quantization.
2. **Early QAT at 50% threshold**: standard late QAT (threshold 0.10) gives ~300 steps of adaptation. Our threshold 0.50 gives ~1,700 steps — critical for int5 where the model needs more time to adapt to only 32 quantization levels.

## Lineage

Built on the SOTA stack from PR #315 (@jfprincz): XSA, Partial RoPE, SmearGate, BigramHash, OrthoInit, U-Net skips, FA3.
