# Non-record: SP8192 + SOTA recipe on 1xA100 — 1.0704 BPB (TTT) / 1.0727 (sliding)

**Author:** Huanyi Xie (`xiehuanyi`)
**Date:** 2026-04-11
**Track:** `non_record_16mb`
**Result:** **val_bpb = 1.07034733** (int6 GPTQ + Brotli + sliding window eval s64 + Legal Score-First TTT)

## TL;DR

This runs the **exact PR #1493 SOTA recipe** (SP8192 + 3-layer recurrence + parallel residuals + QK-gain 5.25 + legal score-first TTT + MuonEq-R + SDClip GPTQ + Brotli + byte shuffle) on **1 × A100 80GB for 4 hours** instead of the required 8 × H100 for 10 minutes. The compute budget is roughly equivalent (~80 H100-minute-equivalent), but because it wasn't actually run on the required hardware, this is a non-record submission.

**Headline result:**
- **TTT BPB: 1.07035** (beats upstream main-leaderboard TTT SOTA 1.0810 by 0.01065)
- **Sliding BPB: 1.07266** (beats upstream main-leaderboard sliding SOTA 1.0827 by 0.01004)
- **Total submission size: 16,019,227 bytes** (under 16 MiB = 16,777,216)

## Why non-record

This submission does **not** qualify for `track_10min_16mb` because:
1. Ran on **1×A100 for 4h (14,400s)** instead of **8×H100 for 10 min**
2. A100 doesn't support FlashAttention-3 (Hopper-only); uses PyTorch SDP with the flash backend as a fallback
3. Never verified on actual 8×H100 hardware

Rough compute comparison:
- H100 BF16: ~990 TFLOPS × 8 × 10 min = ~80 H100-minute-equivalent
- A100 BF16: ~312 TFLOPS × 1 × 240 min = ~76 A100×FLOPs × 3.17 = ~240 A100-minute, approximately matching H100 raw throughput, but without the FA3 speedup.

So this submission is **compute-equivalent** to the main-leaderboard budget, just not on the required hardware.

## What's in the recipe

The training script is a minor adaptation of the PR #1493 script (decompressed from its LZMA+base85 wrapper) with two changes:

1. **FA3 → FA2/SDP fallback**: On A100, FlashAttention-3 is unavailable, so the attention wrapper falls through to PyTorch's `scaled_dot_product_attention` with the flash backend. A manual GQA head-repeat is added for the SDP path since PyTorch SDP doesn't natively support `num_heads != num_kv_heads`.
2. **Python 3.9 compatibility**: Removed `zip(strict=True)` and nested f-string quotes.
3. **`GRAD_ACCUM_STEPS` env override**: Added so the script can be run with arbitrary grad-accumulation on single-GPU setups (default still `8 // world_size`).

Everything else is exactly as in the SOTA submission:
- **SP8192** tokenizer (retokenized FineWeb 10B with a 8192-vocab SentencePiece BPE model borrowed from the 74M_Ternary record)
- **11L × 512d × 8H / 4KV GQA**, MLP 4×, LeakyReLU(0.5)²
- **Depth Recurrence**: loops physical layers 3-5 twice, creating 17 virtual layers from 11 physical, activated at `frac=0.35` of training
- **Parallel Residuals** from layer 7+ (last 4 layers only, GPT-J style)
- **QK-Gain init = 5.25** (per-head learnable query scaling, non-default SOTA setting)
- **Skip Gates** (sigmoid-gated U-Net skip connections)
- **MuonEq-R**: row-normalized Muon, Newton-Schulz 5 steps (plus AdamW for embeddings/scalars/head)
- **Partial RoPE (16/64)** + LN Scale
- **EMA decay 0.9965** with warmdown fraction 0.72
- **MUON_WD = 0.095, ADAM_WD = 0.02, EMBED_WD = 0.085, MATRIX_LR = 0.022**
- **GPTQ with SDClip**: int6 attention/MLP (k=12.85), int8 embeddings (k=20.0), block size 128
- **Brotli-11 + byte shuffle** compression
- **Legal Score-First TTT**: SGD lr=0.005 momentum=0.9, 3 epochs per 32K-token chunk, cosine LR decay, score-before-update ordering

## Numbers (seed 1337)

| Metric | Value |
|---|---|
| Pre-quantization post-EMA BF16 | 1.07610 |
| Int6 quantized (no sliding) | 1.08950 |
| **Int6 + Sliding Window s64** | **1.07266** |
| **Int6 + Sliding + Legal TTT** | **1.07035** ← reported |
| Steps trained | 6371 / 20000 (wallclock capped) |
| Step avg | ~2260 ms (on 1×A100, SDP backend) |
| Peak GPU memory | 41.8 GiB |
| Model params | 35,944,536 |
| Artifact bytes (int6 + brotli) | 15,970,123 |
| Code bytes (uncompressed) | 49,104 |
| **Total submission bytes** | **16,019,227** |

## Comparison vs. upstream records

| Submission | Sliding BPB | TTT BPB |
|---|---|---|
| **This (exp62, 1xA100 4h)** | **1.07266** | **1.07035** |
| PR #1493 SOTA (8xH100 10min) | 1.0827 | 1.0810 |
| PR #1477 (SP8192 + ParResid + TTT) | 1.082~ | 1.0822 |
| PR #1413 (SP8192 + QK5 + TTT) | 1.084~ | 1.0828 |
| PR #1412 (SP8192 + ParResid + SDClip) | 1.086~ | 1.0835 |
| PR #1394 (SP8192 + GPTQ Emb + SDClip) | 1.088~ | 1.0856 |

Delta vs. PR #1493 SOTA: **-0.01004 sliding, -0.01065 TTT**.

## Comparison with exp60 / exp61 (same training config, different QK_gain)

Three runs were made with identical seeds/hyperparams except `QK_GAIN_INIT`:

| Run | QK_GAIN | Int6 | Sliding | TTT |
|---|---|---|---|---|
| exp60 | 5.0 (SOTA default) | 1.09031 | 1.07345 | 1.07137 |
| exp61 | 5.0 + TTT flag at train | 1.09031 | 1.07345 | 1.07137 |
| **exp62** | **5.25** | **1.08950** | **1.07266** | **1.07035** |

QK_GAIN_INIT=5.25 (the SOTA record's exact value, non-default) consistently helps all three quantization/eval phases, matching the SOTA paper's observation that "monotonic improvement from 4.0 to 5.25" holds.

## Reproduction

```bash
pip install brotli sentencepiece
# A100: flash_attn (FA2) optional, falls back to SDP if not installed
# pip install flash-attn --no-build-isolation

# 1. Download docs and retokenize with SP8192 (one-time, ~2h on CPU)
python data/download_hf_docs_and_tokenize.py \
  --repo-id willdepueoai/parameter-golf \
  --remote-root datasets \
  --output-root data \
  --tokenizer-config data/tokenizer_specs_sp8192.json \
  --skip-byte \
  --reuse-sp-model 8192=<path_to_fineweb_8192_bpe.model>

# 2. Train (4h on 1x A100 80GB)
DATA_DIR=./data/ \
SEED=1337 \
VOCAB_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=14400 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Caveats

- **Single seed (1337) only.** A 3-seed mean (e.g. 42, 314, 999) has not been run. The main-leaderboard SOTA reports 3-seed mean/std; this submission is single-seed for time reasons.
- **Non-record hardware.** Not verified on 8×H100; used 4h on 1×A100.
- Two earlier runs (exp60, exp62) crashed with SIGSEGV at the end of their own eval pipelines (torch.compile recompile issue when creating a fresh GPT instance for eval after training). The same saved quantized artifacts were then evaluated successfully via a standalone `eval_only.py` script. The reported numbers come from the standalone eval.
- The `grad_accum=2` variant (exp63/64) OOM'd: the SOTA model with MLP 4× + depth recurrence has a per-micro-batch footprint larger than the simpler v2_full_stack model from earlier rounds.

## Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` — A100-adapted SOTA script (FA3→SDP fallback, Python 3.9 compat, GRAD_ACCUM_STEPS env override)
- `final_model.int6.ptz` — 15.97 MB int6+brotli quantized artifact
- `train_seed1337.log` — full training log
- `eval_seed1337.log` — standalone eval log (sliding + TTT numbers)
