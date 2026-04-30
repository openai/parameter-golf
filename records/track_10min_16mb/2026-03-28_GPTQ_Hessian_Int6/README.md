# GPTQ Hessian Int6

**Builds on**: PR #549 (LeakyReLU² + Legal TTT + Parallel Muon, score 1.1194)

## Summary

Single targeted change: replace the **GPTQ-lite** clip-percentile quantizer with
**full GPTQ** (Frantar et al., 2022) for the int6 weight matrices.

The existing GPTQ-lite minimises the naive weight-space MSE:

```
min ||W - Q(W)||_F^2
```

Full GPTQ minimises the **activation-space** MSE using the input Hessian
`H = X^T X / n` collected from training data:

```
min ||W X - Q(W) X||_F^2
```

This is achieved by quantizing one column at a time and propagating each
column's quantization error to the remaining columns via `H^{-1}`, so that
later weights compensate for earlier rounding errors.

## What Changed

Three additions to `train_gpt.py` on top of the SOTA stack:

1. **`_quantize_int6_lite`** — the original 5-percentile sweep, extracted
   as a named fallback.

2. **`quantize_int6_per_row(t, clip_range, hessian=None)`** — extended to
   accept an optional Hessian. When provided, runs the column-wise GPTQ loop
   on GPU; falls back to `_quantize_int6_lite` otherwise.

3. **`collect_hessians(model, train_files, n_calib_tokens=65536, ...)`** —
   registers forward hooks on every Block's `attn_norm` and `mlp_norm`
   outputs to accumulate `H = X^T X` for Q/K/V projections and MLP-up
   projections. Runs ~65k calibration tokens from training data (not
   validation) in `torch.inference_mode`, adding roughly 3–5 seconds to
   the post-training serialization phase.

Hessians are collected for: `blocks.{i}.attn.c_q/c_k/c_v.weight` and
`blocks.{i}.mlp.fc.weight` (total ~44 matrices for 11 layers). The
attention output projection and MLP down projection fall back to GPTQ-lite
(their inputs require deeper hooks into the forward pass).

## Why This Should Help

- GPTQ-lite sweeps 5 percentile thresholds and picks the one with lowest
  weight-space MSE. It has no knowledge of which input directions are
  important.
- Full GPTQ uses the actual input covariance to weight the quantization
  error. Channels with large activations get better reconstruction. This
  typically cuts int6 reconstruction error by 20–40% with zero inference
  overhead and zero extra bytes in the artifact.
- The model already uses late QAT (int6 fake-quantization during warmdown),
  so weights are already shaped for int6. GPTQ improves the final rounding
  step on top of QAT-aware weights.

## Architecture (unchanged from SOTA)

- 11 layers, 512d, 8 heads, 4 KV heads
- 3× MLP with LeakyReLU(0.5)² activation
- BigramHash (2048 buckets, 128 dim)
- XSA on last 4 layers
- Partial RoPE (16/64 dims)
- LN Scale 1/√(layer+1)
- EMA (decay=0.997) + SWA (every 50 steps)
- GPTQ int6 + lzma level 6 compression
- Legal score-first TTT (SGD lr=0.002, momentum=0.9, 3 epochs/chunk)
- Parallel Muon (batched Newton-Schulz, WD=0.04)
- Warmdown 3500 iters, late QAT at LR scale < 0.15

## Running

```bash
RUN_ID=gptq_hessian_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-28_GPTQ_Hessian_Int6/train_gpt.py \
  2>&1 | tee logs/gptq_hessian_seed1337.txt
```

The `gptq:collecting calibration hessians` log line appears at the end of
training, just before quantization. Expect it to take ~3–5 seconds.

## Expected Score

Target: below 1.1194 BPB (current SOTA).
Estimated gain: −0.003 to −0.008 BPB based on typical GPTQ improvements
over naive per-row quantization at 6 bits.
