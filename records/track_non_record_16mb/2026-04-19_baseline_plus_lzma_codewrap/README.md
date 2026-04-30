# SP8192 Baseline Reproduction with LZMA Code-Wrap

**Track:** non_record_16mb
**Date:** 2026-04-19
**val_bpb:** 1.08814 (seed 1337, sliding-window quantized)
**Total submission:** 15,988,151 bytes (15.988 MB, under 16 MB cap with 11,849 bytes of headroom)

## What this submission is

This is a single-seed baseline reproduction of the SP8192 + int6 GPTQ stack (PR #1394 line), with the `--compress` LZMA code-wrap applied to the `train_gpt.py` source. It is explicitly **not a novel submission** — the training stack uses techniques already on the leaderboard. Included here as a reference point and to validate our experimental harness.

## Architecture / hyperparameters

- **Tokenizer:** SP8192
- **Model:** 11 layers, model_dim=512, 8 heads, 4 KV heads (GQA), MLP mult=4 (SwiGLU)
- **Depth recurrence:** loop_start=3, loop_end=5, num_loops=2 (3-layer recurrence, enabled at 50% training)
- **Parallel residuals:** from layer 7+
- **Activation:** LeakyReLU(0.5)²
- **QK-Gain init:** 5.25
- **Optimizer:** Muon on matrices (polar-express Newton-Schulz, backend_steps=5), Adam on scalars/embeds; momentum warmup 1500 steps
- **Weight decay:** 0.095 on Muon + embed, 0.02 on Adam
- **EMA decay:** 0.9965
- **Schedule:** warmup 20 steps, warmdown_frac=0.3, stop_on_wallclock=600s (−12s GPTQ reserve)

## Quantization stack

- **GPTQ** with full-Hessian Cholesky error compensation (64 calibration batches)
- **SDClip** per-row clipping: `s = k × std(row)`, k=12.85 for int6 matrices, k=20 for int8 embeddings
- **Matrix bits:** int6 on attn (c_q, c_k, c_v, proj), int6 on mlp (fc, proj)
- **Embedding bits:** int8 on tok_emb
- **Small tensors / scales:** float16 passthrough (control tensors, skip gates, etc.)
- **Compression:** byte-shuffle stride 2 → Brotli-11
- **Source code:** LZMA-compressed via `--compress` flag, yielding an 18,447-byte bootstrap that `exec`s the original source

## Hardware and runtime

- **Hardware:** 8×H100 80GB SXM on RunPod
- **PyTorch:** 2.9.1+cu128
- **Steps completed:** 4870 / 20000 (wallclock-capped)
- **Training time:** 588,018 ms
- **ms/step:** 120.74
- **Peak VRAM:** 39.0 GiB

## Results

| Eval | Val Loss | val_bpb |
|---|---|---|
| pre-quantization post-ema | 2.8301 | 1.0956 |
| quantized (non-sliding) | 2.8541 | 1.1049 |
| quantized_sliding_window | 2.8108 | **1.0881** |

## Compliance

- ✅ Training ≤600s wallclock (588,018 ms)
- ✅ Artifact ≤16.0 MB (15,988,151 bytes)
- ✅ Quantized eval runs in-bound
- ✅ No SLOT, no pre-quant TTT, no ETLB, no n-gram cache
- ❌ No score-first TTT (this submission is pre-TTT stack)
- ❌ Single seed only (1337) — not eligible for record track

## Attribution

This submission combines techniques from community PRs:
- SP8192 + GPTQ + SDClip: @clarkkev (PR #1394)
- Depth recurrence: @dexhunter (PR #1331, #1437)
- Parallel residuals: @Robby955 (PR #1412), @msisovic (PR #1204)
- LeakyReLU²: @abaybektursun (PR #549)

No novel techniques claimed.

## Files

- `train_gpt.py` — variant run016v (1403 lines)
- `train_seed1337.log` — training log
- `submission.json` — metadata
