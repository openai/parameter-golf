# 2026-03-21. Mixed Quantization + BigramHash + SWA

First submission. Configuration assembled the night before the deadline from minimally-proven features after the `020_ultimate` failure. An honest working baseline, not a winning number.

## Result

| Metric | Value |
|---|---|
| val_bpb post-roundtrip | **1.2421** |
| val_bpb pre-roundtrip | 1.1924 |
| Quantization gap | 0.0497 |
| Artifact size | 13 279 428 bytes (13.28 MB) |
| Steps in 600 seconds | 11 070 |
| Step time | 54.20 ms |
| SWA snapshots | 115 |
| GPU | 8× H100 80GB SXM |

## Architecture

| Component | Setting |
|---|---|
| Layers | 10 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 (GQA) |
| MLP | ReLU² with 3x expansion, hidden=1536 |
| vocab | 1024 BPE (SentencePiece) |
| tied embeddings | yes, fp16 export |
| BigramHash | 10 240 buckets × 128 dim projected to model_dim |
| U-Net skips | yes, across layers with sigmoid gate |
| train_seq_len | 1024 |

## Optimizer

Muon with weight decay 0.04 and gradient clipping 0.3. Momentum warmup from 0.85 to 0.99 over 1500 steps.

| Group | LR | Settings |
|---|---|---|
| Bank matrix weights (Muon) | 0.02 | momentum=0.99, WD=0.04 |
| Scalars (Adam) | 0.04 | betas=(0.9, 0.95) |
| Tied embeddings | 0.05 | betas=(0.9, 0.95) |

Warmdown: 1500 steps until the end of training, linear LR decay to 10% of peak.

## Quantization and compression

Mixed precision scheme:

| Weight type | Bit rate |
|---|---|
| Attention (Q, K, V, output) | INT6 |
| MLP weights | INT6 |
| Token embeddings | INT8 |
| Scalars (LN weights, biases) | fp16 passthrough |

Straight-Through Estimator is active from the first training step. On forward all weights get quantized, on backward the quantization is ignored. The model learns to handle quantization from the start, not at the end.

Final artifact compression: zstd level 22. Ratio about 3.82x vs packed bytes.

## Stochastic Weight Averaging

SWA start: step 5335 (about 50% of 11 070 steps). Period: every 50 steps. Total snapshots: 115.

Final weights come from averaging those snapshots in float32, then quantizing. Why it matters: averaging gives a flatter weight-space optimum, and a flat optimum handles quantization noise better.

Without SWA: quantization gap about 0.08 bpb.
With SWA: quantization gap 0.05 bpb.

## Honest assessment

This submission isn't competitive with the leaderboard top (1.1428 bpb on March 20). The 1.2421 number is the outcome of a single production run on H100 assembled in a couple of hours between the `020_ultimate` failure and the RunPod instance deadline.

I tried a more ambitious configuration called `020_ultimate` (12 layers of SwiGLU, XSA, window attention, RoPE base=50000, EMA, label smoothing), got 1.4143, which is below the organizers' baseline. Full breakdown of why in [docs/EXPERIMENTS.md](../../../docs/EXPERIMENTS.md). Quickly rolled back to a minimally-working stack without SwiGLU and window attention, it gave 1.2421 in 600 seconds.

Claims of "I could've done better" are speculation from my side: I don't have saved training logs for other configurations. What I actually did in that 24-hour window is captured by logs: baseline on 3090, the failed 020_ultimate, this submission. The April run at 1.1431 came ten days later and was done calmly.

## What to try next time

1. **More data.** I used 10 train shards due to disk constraints on RunPod. 80 shards (a full FineWeb sample) would likely add another 0.01 to 0.02 bpb.

2. **Per-channel quantization.** Currently per-row. Per-channel gives a finer scale-factor calibration, the quantization gap should drop to 0.02 to 0.03.

3. **Smoother STE schedule.** STE is on from step 0 here. Late QAT (activating STE only in the last 30% of training) worked better in my later experiments.

4. **GPTQ instead of STE.** One-shot post-training quantization with Hessian-aware error compensation. In my April submission it worked.

## Files

- `train_gpt.py`: full training script, 864 LoC
- `train.log`: training log with all val_bpb measurements
- `submission.json`: leaderboard metadata
- `README.md`: this file

## Running it

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requirements: 8× H100 with 80 GiB memory, CUDA 12.8+, PyTorch 2.8+.

## Pull Request

[openai/parameter-golf#370](https://github.com/openai/parameter-golf/pull/370), opened 2026-03-21.
