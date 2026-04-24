# 2026-04-01. Turbo-Muon + EngramLite + VE(8,9,10)

Second submission. Made calmly, ten days after the `020_ultimate` failure, on top of the well-documented PR #1089. Main goal: a clean reproducible result with 3-seed verification, no chasing the third decimal.

## Result

**val_bpb: 1.1431** (mean across three seeds, std=0.0007)

| Seed | step_avg | steps | val_bpb sliding | val_bpb roundtrip | Artifact |
|---|---|---|---|---|---|
| 1337 | 106.74 ms | 5 538 | **1.1425** | 1.1657 | 15 988 293 |
| 42 | 106.09 ms | 5 572 | **1.1438** | 1.1669 | 15 978 184 |
| 2024 | 106.00 ms | 5 576 | **1.1431** | 1.1652 | 15 985 158 |
| **Mean** | **106.28 ms** | **5 562** | **1.1431** | **1.1659** | |

Hardware: 8× H100 80GB SXM. Time: 591 seconds. Quantization gap: 1.1659 roundtrip minus 1.1431 sliding = **0.0228**. A substantial improvement over my first submission (gap was 0.0497).

## Context

Base: [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @Bortlesboat, Turbo-Muon + EngramLite stack with Parameter Banking and GPTQ Mixed-Precision. One of the cleanest public PRs in upstream, with a thorough README, reproducible config, and documented trade-offs.

Instead of trying to invent another merkl-stack of techniques from scratch, I took a good base and tuned seven hyperparameters. Humbler approach, cleaner result.

## Deltas to PR #1089

| Parameter | PR #1089 | Mine | Reason |
|---|---|---|---|
| `MATRIX_LR` | 0.025 | **0.030** | Faster convergence in 600 s |
| `SCALAR_LR` | 0.025 | **0.030** | Matched to matrix_lr |
| `WARMDOWN_ITERS` | 3500 | **4500** | Smoother weight averaging in the tail |
| `MUON_MOMENTUM_WARMUP_STEPS` | 1500 | **1000** | Reach target 0.99 sooner |
| `VE_LAYERS` | 9, 10 | **8, 9, 10** | Extra token identity on the middle layer |
| `NGRAM_BUCKETS` | 8192 | **10240** | Wider n-gram coverage |
| `NGRAM_DIM_PER_HEAD` | 32 | **48** | Denser n-gram embedding |

## Architecture

| Component | Setting |
|---|---|
| Layers | 11 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 (GQA) |
| MLP | 3.5x with LeakyReLU(ASQU v3 per-layer slopes)² |
| XSA | All 11 layers |
| EngramLite | 2 heads × 2 orders (bigram + trigram), **10240 buckets**, **48 dim/head** |
| U-Net skip connections | sigmoid-gated |
| RoPE | Partial, 16 of 64 dims |
| LN Scale | 1/√(layer+1) |
| Logit Softcap | 30.0 |
| ValueEmbedding | **Layers 8, 9, 10** |
| SmearGate | causal shift blending |
| Embeddings | tied input/output |
| Vocab | 1024 BPE |
| train_seq_len | 2048 |

## Optimizer

| Group | LR | Settings |
|---|---|---|
| Bank weights (Turbo-Muon) | **0.030** | momentum=0.99, WD=0.04, NS=4, post_norm=row_col |
| Embeddings (Adam) | 0.6 | betas=(0.7, 0.95), WD=0.04 |
| Head / tied embed (Adam) | 0.035 | betas=(0.7, 0.95) |
| Scalars (Adam) | **0.030** | betas=(0.9, 0.95) |

Turbo-Muon is a Muon variant with parallel communication: reduce-scatter of local gradients by weight banks, local Newton Schulz (NS=4 instead of 5), all-gather of results. Saves about 15% of step time vs standard Muon + all-reduce.

## Quantization

Main method: GPTQ with Hessian-aware Cholesky error compensation. I reserve 9 seconds of the 600-second budget for calibration.

Dynamic mixed-precision:
- Base rate: int5 for all 66 weight groups (attention Q, K, V, out; MLP up, down; across all layers)
- Promotion: 0 groups promoted to int6 or int7

Why int5 everywhere. The extended EngramLite (10240 × 48) and VE on 3 layers added about 950K parameters. The model ends up at 31.6M parameters vs 30.7M in PR #1089. Pre-compression size hit 16.36 MB, above budget. This forced all weights into int5 without promotion and required selective pruning.

Selective pruning: 20.5% of ±1, ±2 values (the smallest in magnitude after quantization) get zeroed. It's a recursive process: pruning iterates until the Brotli-11 compressed size fits 16 MB minus the code size.

Final compression: Brotli level 11 with byte-shuffle preprocessing. Byte-shuffle regroups the int array: high bytes first, then low bytes. This makes the sequence more uniform, Brotli compresses it 3 to 5% better.

Late QAT: regular fp32 training up to a threshold step, then soft-round sigmoid alpha ramp with threshold=0.15. The model has full freedom in fp32 space for the first 85% of training and smoothly adapts to quantization in the last 15%.

## Weight Averaging

SWA: float32 accumulation every 50 steps after warmdown threshold. Ended up with 18 snapshots over 591 seconds. Averaged uniformly at the end of training.

EMA: decay=0.997, applied in addition to SWA. EMA smooths across all training with bias toward late steps, SWA gives a flat optimum in the tail. Combining the two gives robustness under quantization.

## Observations from this run

Extended EngramLite and VE on 3 layers increase parameter count, which pushes everything to aggressive int5 with selective pruning. A trade-off: wider n-gram coverage and extra token identity give more capacity, but hard quantization eats part of the win.

The sweet spot for EngramLite in my configuration is probably closer to 8192 × 32 (PR #1089 default), which would allow promoting sensitive groups to int6 and skip pruning. I didn't get to verify this within the scope of this submission.

Quantization gap 0.023 is noticeably better than my March submission (0.05). Combination of Late QAT + SWA + EMA + Brotli works.

## Running it

Full verification needs three runs:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42   torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2024 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Mean val_bpb should match 1.1431 within 0.001.

Requirements: 8× H100 80 GiB SXM, CUDA 12.8+, PyTorch 2.8+. Pipeline uses flash_attn v3 (Hopper-specific) with SDPA fallback.

## Files

- `train_gpt.py`: full training script, 146 041 bytes
- `train_seed1337.log`, `train_seed42.log`, `train_seed2024.log`: logs for all three runs
- `submission.json`: metadata with deep stats
- `README.md`: this file

## Pull Request

[openai/parameter-golf#1205](https://github.com/openai/parameter-golf/pull/1205), opened 2026-04-01.

## Credits

- **Base recipe**: [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @Bortlesboat, Turbo-Muon + EngramLite + Parameter Banking + GPTQ Mixed-Precision
- **LeakyReLU²**: [PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #518](https://github.com/openai/parameter-golf/pull/518)
- **XSA**: [PR #265](https://github.com/openai/parameter-golf/pull/265), [PR #287](https://github.com/openai/parameter-golf/pull/287)
- **SmearGate + BigramHash**: [PR #198](https://github.com/openai/parameter-golf/pull/198)
- **Polar Express coefficients**: Amsel et al. (arXiv:2505.16932)
- **GPTQ approach**: [PR #634](https://github.com/openai/parameter-golf/pull/634)
