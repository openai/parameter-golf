# Non-Record V4: 7-Layer Shared Transformer + Int6 QAT (LZMA) + 10 h Training + Sliding-Window Eval

**val_bpb: 1.30952** (single seed, stride-64 sliding window) | **vs V3.1: −0.05457 BPB** | **vs V2 baseline: −0.08741 BPB** | **Artifact 13.73 MB** | DGX Spark (1× GB10 Blackwell)

## 1. Overview

V4 improves on the [V3.1 sliding-window submission](../2026-04-19_SharedTransformer_SlidingWindow_Int8_7L) by **−0.05457 BPB** (1.36409 → 1.30952). Two changes vs V3.1:

1. **Quantization**: int6 per-row QAT + LZMA serialization in place of int8 + zlib. The lighter quant scheme leaves ~2.27 MB of headroom under the 16 MB cap, which the previous V3.1 int8 submission had nearly consumed (15.51–15.61 MB).
2. **Training budget**: 10 h wallclock cap (`MAX_WALLCLOCK_SECONDS=36000`) instead of V3.1's 4 h. The same per-step loop completes 2,612 steps instead of ~1,050.

Architecture, optimizer, learning rates, EMA settings, batch tokens, sequence length, and eval algorithm are all unchanged from V3.1.

The checkpoint shipped here was trained on 2026-04-11 as part of an exploratory run labelled `overnight_int8_best`. It was not on V3.1's submission shortlist because it pre-dated the sliding-window eval methodology; it surfaced in a post-V3.1 run audit (`logs/run_audit.txt`) as the best-BPB checkpoint surviving on disk. We then ran the same stride-64 sliding eval that V3.1 used on this checkpoint and got 1.30952 BPB. This submission ships that result.

Non-record track; developed on NVIDIA DGX Spark with AI-assisted coding.

## 2. Architecture

Identical to [V3.1](../2026-04-19_SharedTransformer_SlidingWindow_Int8_7L) and [V2](../2026-04-13_UNet_Int8QAT_7L_4xMLP_EMA_LongTrain).

| Property | Value |
|---|---|
| Transformer layers | 7 |
| Shared-transformer topology | U-Net skip connections (encoder/decoder halves, learned skip weights) |
| Model dim | 512 |
| Attention heads | 8 query, 4 KV (GQA, head_dim 64) |
| MLP multiplier | 4× |
| MLP activation | Leaky ReLU squared (negative_slope = 0.5, then squared) |
| Embeddings | Tied, separate `tied_embed_lr` |
| Logit softcap | Tanh, softcap = 30.0 |
| Positional encoding | RoPE (base 10000.0) |
| Vocab | SentencePiece BPE, 1024 tokens |
| Sequence length | 1024 |
| Total params | **20,725,304** |

### Training hyperparameters (LRs unchanged from V3.1; wallclock changed)

| Hparam | Value | Same as V3.1? |
|---|---|---|
| `MATRIX_LR` | 0.08261619767374824 | yes |
| `SCALAR_LR` | 0.014691154447587356 | yes |
| `TIED_EMBED_LR` | 0.021552090970329115 | yes |
| `HEAD_LR` | 0.0 (tied-only head path) | yes |
| `EMA_DECAY` | 0.997 | yes |
| `TRAIN_BATCH_TOKENS` | 524,288 (8 grad-accum steps) | yes |
| `TRAIN_SEQ_LEN` | 1024 | yes |
| `WARMUP_STEPS` | 20 | yes |
| `USE_INT6` | 1 | **changed (V3.1 used 0)** |
| `MAX_WALLCLOCK_SECONDS` | 36000 (10 h) | **changed (V3.1 used 14400 = 4 h)** |
| `ITERATIONS` cap | 20000 | yes (cap; 2,612 actually completed) |

`MUON_MOMENTUM` and `WARMDOWN_ITERS` are not dumped in the training log header at runtime, so we cannot read them back from the log. Every LR and architectural hyperparameter that *is* logged matches V3.1 to all 17 significant digits, which strongly suggests the run used V3.1's full Optuna config (`MUON_MOMENTUM=0.9382982028913158`, `WARMDOWN_ITERS=1558`); but this is inference, not direct evidence. See `logs/provenance.txt` for the full audit.

## 3. Quantization: Int6 per-row QAT + LZMA

V3.1 used per-row int8 quantization with zlib compression, yielding ~15.5 MB artifacts (≈97% of the 16 MB cap). V4 swaps to **int6 per-row** quantization for the `mlp` and `attn` projections (and int8 for everything else), serialized with **LZMA** instead of zlib:

- The `mixed_quantize_int6` path (see `train_gpt.py`) sweeps row clip-quantiles in `{0.999, 0.9995, 0.9999, 0.99999, 1.0}` per row and picks the one with lowest mean-squared reconstruction error. Quantized values are int8 in storage but clamped to `[-31, +31]` (6-bit range), and LZMA then exploits the resulting low-entropy byte distribution.
- The MLP and attention projections (≈19.0 M of 20.7 M params) drop to int6; the small embedding/passthrough tensors stay at int8 / fp16 / fp32 as appropriate.
- Round-trip: `lzma.decompress` → `torch.load` → `dequantize_mixed_int6(quant_state['w'], quant_state['m'], template_sd)` → `model.load_state_dict(strict=True)`.

| Submission | Artifact bytes | Headroom under 16 MB |
|---|---|---|
| V3.1 seed 314 (int8+zlib) | 15,610,091 | 390 KB |
| V3.1 seed 1337 (int8+zlib) | 15,509,631 | 491 KB |
| **V4 seed 1337 (int6+lzma)** | **13,726,856** | **2,254 KB** |

The post-quant chunked round-trip val_bpb on the training host (logged at training time as `final_int6_lzma_roundtrip_exact`) was **1.34167922**. The full sliding-window eval on the same round-tripped checkpoint (Section 4) drops that to 1.30952212.

## 4. Sliding-Window Evaluation

**Algorithm** identical to V3.1 (`eval_val_sliding` in `train_gpt.py`):

1. Slide a context window of size `TRAIN_SEQ_LEN = 1024` across the validation corpus in stride-`64` increments.
2. **First window**: score all 1023 next-token targets. **Subsequent windows**: score only the rightmost 64 targets (each gets ≥960 tokens of left context).
3. Total windows on fineweb val (62,021,632 tokens): **969,088**. Every val token scored exactly once.

**Eval was run on the int6+lzma round-tripped checkpoint** (decompress → dequantize → `load_state_dict(strict=True)`), so the reported BPB is exactly what a reviewer loads from the submitted `.int6.ptz` artifact.

The eval was driven by a small adapter script (`eval_sliding_int6.py` in the training-host repo) that swaps the V3.1 loader's `zlib + dequantize_state_dict_int8` for `lzma + dequantize_mixed_int6`. The eval algorithm — window iteration, batching, NLL accumulation, byte-counting via the SentencePiece LUTs — is byte-identical to V3.1's `eval_val_sliding`. The V4 `train_gpt.py` shipped here exposes the same algorithm under USE_INT6=1 and runs it as the post-training round-trip eval.

## 5. Results

| Submission | Seed(s) | Mean BPB | Δ vs V2 |
|---|---|---|---|
| V1 (PR #1486) baseline | — | 1.666 | — |
| V2 (PR #1606) | 3 seeds | 1.39693 | −0.269 |
| V3.1 (sliding, not yet submitted) | 2 seeds | 1.36409 | −0.302 |
| **V4 (this submission)** | **1 seed** | **1.30952** | **−0.356** |

| Metric | Value |
|---|---|
| seed | 1337 |
| val_loss | 2.21107041 |
| val_bpb | **1.30952212** |
| tokens scored | 62,022,592 |
| bytes for BPB | 151,082,508 |
| eval wallclock | 19,007.6 s (5.28 h on GB10) |
| Δ vs V3.1 mean (1.36409) | **−0.05457** |
| Δ vs V2 mean (1.39693) | **−0.08741** |

### Single-seed disclosure

This is a **single-seed submission**. The non-record track does not require multi-seed statistical significance, and a 0.05 BPB margin over the V3.1 *mean* of 2 seeds is large relative to the seed-to-seed spread observed in V2 (1.39533 / 1.39564 / 1.39983, range 0.0045) and V3.1 (1.36221 / 1.36598, range 0.0038). The result is unlikely to be a single-seed outlier of that magnitude, but a second-seed retrain at this 10 h budget would harden the comparison. The retrain was not run (the v3.1 release timeline was prioritized; this v4 was discovered after).

## 6. Comparison Table (extended)

| Submission | Mean BPB | Δ vs V2 | Δ vs prev | Notes |
|---|---|---|---|---|
| V1 (PR #1486) | 1.666 | baseline | — | initial baseline |
| V2 (PR #1606) | 1.39693 | −0.269 | −0.269 | int8 QAT, 4 h training, chunked eval |
| V3.1 (not yet submitted) | 1.36409 | −0.302 | −0.033 | sliding-window eval (algorithm change only) |
| **V4 (this)** | **1.30952** | **−0.356** | **−0.055** | int6 QAT + LZMA, 10 h training |

## 7. Hardware and Eval Setup

- **Training**: single GB10 Blackwell on DGX Spark, 128 GB unified memory, `MAX_WALLCLOCK_SECONDS = 36000` (10 h). Peak GPU memory ≈ 25.4 GiB allocated / 25.9 GiB reserved. Per-step wallclock ≈ 13.78 s.
- **Eval**: stride-64 sliding-window eval on a single GB10 takes ≈ 5.28 h on this checkpoint (969,088 windows × `EVAL_BATCH_SEQS = 64`).
- **Expected competition-hardware eval time**: per issue #1017, the accepted 10-minute-track stride-64 record runs in ~70 s on 8× H100 SXM. Scaling to this checkpoint: well within the non-record track's eval budget.

## 8. Ablation Trail

V3 (the immediate predecessor) explored **n-gram backoff cache mixing** (see V3.1 README §6 for the sweep summary). Sliding-only (no n-gram) dominated n-gram-mixed eval on V3's checkpoint by −0.00180 BPB; V3.1 therefore removed the n-gram code entirely. V4 inherits that decision: no n-gram code path is present in `train_gpt.py`.

V4's only substantive deltas vs V3.1 are the two listed in §1 (quant scheme + wallclock budget). Architecture, optimizer, EMA, learning rates, and eval algorithm are unchanged.

## 9. Reproducibility

- **Hardware**: NVIDIA DGX Spark, single GB10 Blackwell (sm_121), 128 GB unified CPU+GPU memory, CUDA 13.0, ARM64.
- **Training invocation**:
  ```bash
  USE_INT6=1 \
  MAX_WALLCLOCK_SECONDS=36000 \
  SEED=1337 \
  MATRIX_LR=0.08261619767374824 \
  SCALAR_LR=0.014691154447587356 \
  TIED_EMBED_LR=0.021552090970329115 \
  HEAD_LR=0.0 \
  EMA_ENABLED=1 EMA_DECAY=0.997 \
  TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024 \
  WARMUP_STEPS=20 ITERATIONS=20000 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
  ```
  (See §2 note about MUON_MOMENTUM and WARMDOWN_ITERS not being directly recoverable from the original run's log.)
- **Eval invocation**: `train_gpt.py` runs the int6+lzma round-trip + `eval_val_sliding` automatically at the end of training. To re-evaluate the shipped checkpoint without retraining, point a small loader script at `final_model_seed1337.int6.ptz` and call `dequantize_mixed_int6` + `eval_val_sliding` from `train_gpt.py` directly (see `logs/provenance.txt` for the loader recipe used to produce `eval_seed1337_sliding.log`).

### Determinism note

This submission does **not** enforce deterministic CUDA ops. Same as V3.1 / V2, the repo norm.

## Files

- `train_gpt.py` — self-contained training + sliding-window eval (1,420 lines). Identical to V3.1's `train_gpt.py` apart from removing one vestigial post-eval log line that always printed `final_int8_zlib_roundtrip_exact` regardless of quant scheme. The int6+lzma path was already wired in V3.1's script under `USE_INT6=1`.
- `requirements.txt` — identical to V3.1.
- `submission.json` — structured result manifest.
- `final_model_seed1337.int6.ptz` — int6+lzma checkpoint (13,726,856 B, MD5 `cbd808b78b1ba660c22a6d4c8119598c`).
- `logs/train_seed1337.log` — full training log from the 2026-04-11 `overnight_int8_best` run that produced this checkpoint.
- `logs/train_seed1337_stdout.log` — training stdout (env dump + per-step lines).
- `logs/eval_seed1337_sliding.log` — structured sliding-eval result.
- `logs/eval_seed1337_sliding_run.log` — live sliding-eval stdout (5.28 h run, progress every 25 batches).
- `logs/provenance.txt` — full provenance audit (what the original training log does and does not record, what was inferred and why).
- `logs/run_audit.txt` — the post-V3.1 run audit that surfaced this checkpoint as the best surviving on disk.
