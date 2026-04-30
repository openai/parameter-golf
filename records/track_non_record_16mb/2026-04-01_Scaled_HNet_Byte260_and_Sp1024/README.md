# [Non-record] Scaled Byte-level H-Net matches 4-hour subword-level baseline (H-Net val_bpb = 1.2070)

>**12L H-Net (3 Encoder + 6 Main + 3 Decoder) + byte260 + GQA KV4 + INT6 GPTQ + QAT + zstd-22 + Stride-64 Sliding Eval; val_bpb: 1.2070**

Follow-up to [PR #1104](https://github.com/openai/parameter-golf/pull/1104), which introduced a byte-level H-Net that learns whitespace-aligned, word-like boundaries from raw bytes but reached only 1.36 BPB on a 4 hours run, significantly worse than the competition's 4 hours baseline of 1.20.

This submission closes that gap. By scaling the architecture from 9 layers (2 outer layers x2; Encoder + Decoder) to 12 layers (3 outer layers x2; Encoder + Decoder), adding INT6 GPTQ quantization, quantization-aware training (QAT), sliding window eval, and a `torch.compile` compatible chunking strategy, the H-Net `byte260` model now reaches **1.2070 BPB** in a 4-hour set-up, matching the 4-hour baseline (1.2074) and a comparable `sp1024` H-Net (1.2107).

We also include a detailed comparison of `byte260` vs `sp1024` H-Net, analysis of how the fixed chunking strategy impacts sliding window eval, and results on dynamic chunking at inference.

## Key Results

#### 4 hour runs

| Config | BPB | Steps | Artifact size |
|--------|-----|------:|--------:|
| **H-Net `byte260` 12L (3 OLs, KV=4)** | **1.2070** | 153,381 | 15.8 MB |
| H-Net `sp1024` 11L (3 OLs, KV=4, div=2) | 1.2107 | 84,278 | 14.6 MB |
|**Baselines**||
| 4-hour baseline | 1.2074 | 329,430 | 15.8 MB |
| H-Net `byte260` 9L ([PR #1104](https://github.com/openai/parameter-golf/pull/1104)) | 1.3595 | 85,242 | 16.0 MB |


#### 10 min runs

| Config | BPB | Steps | Artifact size |
|--------|-----|------:|--------:|
| H-Net `byte260` 12L (3 OLs, KV=4) chunk_size=6 | 1.3123 | 5,957 | 15.2 MB |
| H-Net `sp1024` 11L (3 OLs, KV=4) chunk_size=6 | 1.2754 | 6,161 | 14.1 MB |
|**Baselines**||
| Naive baseline (9L 512dim `sp1024` KV4) | 1.2244 | 13,780 | 15.9 MB |
| H-Net `byte260` 9L OL2 chunk_size=9 ([PR #1104](https://github.com/openai/parameter-golf/pull/1104)) | 1.4116 | 4,520 | 15.8 MB |
| H-Net `sp1024` 9L OL2 chunk_size=12 ([PR #1104](https://github.com/openai/parameter-golf/pull/1104)) | 1.3734 | 4,466 | 16.0 MB |

(*OL = Outer Layers, used for H-Net Encoders and Decoders*)


- At 10 minutes, `byte260` still has a significant gap compared to `sp1024` (+0.037 BPB). With a 4-hour budget, `byte260` closes the gap entirely, matching both the `sp1024` H-Net (1.2070 `byte260` vs 1.2107 `sp1024`) and the 4-hour baseline (1.2074).
- The byte H-Net still has clear optimization headroom as val BPB is still decreasing at the 4 hours cutoff. This suggests that part of the remaining gap is due to optimization budget rather than a hard limit of byte-level H-Net.

## Changes compared to [PR #1104](https://github.com/openai/parameter-golf/pull/1104)

| | PR #1104 | This submission (`byte260`) |
|--|---------|-----------------|
| **Architecture** | 9L, OL2 (2 encoder + 5 main + 2 decoder) | 12L, OL3 (3 encoder + 6 main + 3 decoder) |
| **Parameters** | 17.5M | 23.0M |
| **Quantization** | int8+zlib (no GPTQ) | int6 GPTQ + zstd-22 |
| **QAT** | None | 85% of warmdown (~21k steps) |
| **Evaluation** | Full-sequence roundtrip | Sliding window (stride 64) |
| **Best 4hr BPB** | 1.3595 | **1.2070** |

## Architecture

Same 1-stage H-Net layout as [PR #1104](https://github.com/openai/parameter-golf/pull/1104), scaled up:

```
Input -> Embedding -> Encoder (3 blocks) -> Routing -> ChunkLayer (L -> C)
      -> Main Transformer (6 blocks) -> DeChunkLayer (C -> L)
      -> + Residual Skip -> Decoder (3 blocks) -> LM Head
```

- **12 layers total**: 3 encoder + 6 main + 3 decoder (OL3)
- **512 model dim**, 8 heads, 4 KV heads (GQA)
- **22.97M parameters**, `byte260` tokenizer (vocab=260)
- **Chunk target size**: 6

## Trade-off: Compile-Chunking

- **Issue**: To take full advantage of `torch.compile`'s speed efficiency, this requires tensor shapes to be fixed at trace time. In the H-Net dynamic chunking variant, the number of chunks varies for every batch, depending on how many boundaries the router predicts. This results in a graph break that prevents compilation, making training 2.6x slower (previously 245ms - in [PR #1104](https://github.com/openai/parameter-golf/pull/1104) - vs the current 93ms per step).

- **One solution**: `torch.compile` requires fixed-size tensors, so we decide to chunk the sequence at `L // CHUNK_DIVISOR` positions, where *L* = sequence length.

  This simple solution comes with some issues:

  -   *Issue 1 - Boundary truncation*: When the router predicts more boundaries than the limit, the first `L // CHUNK_DIVISOR` boundaries are kept and everything past the cap gets merged into one large final chunk. In early training (~first 1k steps), truncation can happen. Since truncated boundaries receive no useful gradient, the router learns to simply predict fewer boundaries rather than learning where to place them. Empirically, we didn't find this to affect the quality of the learned boundaries.

  -   *Issue 2 - Some wasted compute*: When the actual boundary count is below the cap (i.e., later in training), the main transformer processes padded positions. In practice, we found this overhead to be negligible.

  - *Potential improvements*: The truncation issue could be mitigated by (a) setting the cap to `L` (no truncation), though this negates much of the compile speedup due to wasted compute on padded positions (i.e., issue 2), or (b) starting with a larger cap (e.g., `L` or `L // 2`) in early training and reducing it to `L // 4` as the router converges, e.g., by tracking boundary count statistics. Approach (b) requires recompilation of the traced graph at each cap change (which is expensive), so it could only be practical for long runs where the recompilation cost is amortized. Given our compute constraints and the competitive results from the fixed-cap approach, we haven't explored alternatives.

- **Choosing CHUNK_DIVISOR**: In [PR #1104](https://github.com/openai/parameter-golf/pull/1104), we observed that after training, the byte-level model converges to an average chunk size of 4-5 bytes (when chunk_target_size=6).

   > For `L=2048` bytes, that's roughly 410-512 chunks. With `CHUNK_DIVISOR=4` the cap is 2048 // 4 = `512`, which should be enough to capture most of the boundaries. Issue 1 only affects early training: the router produces more boundaries in the beginning, and then converges to a lower average chunk size (after ~1k steps).


### 10 min runs

#### `byte260` - `CHUNK_DIVISOR` sweep

All runs: 12L, OL3, 512dim, `byte260`, 8xH100, 10 minutes.

| CHUNK_DIVISOR | Steps | ms/step | Float BPB | Sliding INT6 BPB |
|-----|------:|--------:|----------:|-----------------:|
| 2 | 3,155 | 175.9 | 1.3714 | 1.3348 |
| **4** | **5,957** | **93.2** | **1.3465** | **1.3123** |
| 6 | 7,219 | 76.9 | 1.3494 | 1.3435 |
| 8 | 7,957 | 69.8 | 1.3480 | 1.3273 |

As explained above, given our observation from [PR #1104](https://github.com/openai/parameter-golf/pull/1104) that chunks converge to an average of 4-5 bytes, we choose `CHUNK_DIVISOR=4` during training.

## `byte260` vs `sp1024`

### 10 min runs

Comparing the `byte260` H-Net (this submission) against `sp1024` H-Net runs.

| Config | Tokenizer | Params | BPB (sliding INT6) | Steps | ms/step | Artifact size |
|--------|-----------|-------:|-------------------:|------:|--------:|--------------:|
| 12L OL3 DIV=4 KV4 | `byte260` | 23.0M | **1.3123** | 5,957 | 93.2 | 15.2 MB |
| 11L OL3 DIV=4 KV4* | `sp1024` | 21.5M | 1.2754 | 6,161 | 90.0 | 14.1 MB |
| 11L OL3 DIV=2 KV4* | `sp1024` | 21.5M | 1.2820 | 3,219 | 172.4 | 14.2 MB |

\* For the `sp1024` run, we use 11 layers (11L) instead of 12, as otherwise the `sp1024` model doesn't fit within the 16MB budget.

\* DIV refers to the CHUNK_DIVISOR

> At 10 minutes, `byte260` (1.3123) still has a significant gap compared to `sp1024` (1.2754) - i.e., +0.0369 BPB.


### 4 hours runs

| | `byte260` (12L OL3 KV4) | `sp1024` (div=2, 11L OL3 KV4) | `sp1024` (div=4, 11L OL3 KV4) |
|--|------:|------:|------:|
| CHUNK_DIVISOR | 4 | 2 | 4 |
| Params | 23.0M | 21.5M | 21.5M |
| Pre-QAT BPB | 1.2125 | 1.2120 | 1.1904 |
| Float Roundtrip BPB | 1.2302 | 1.2163 | 1.2118 |
| Float Sliding BPB | 1.1980 (−0.032) | 1.1940 (−0.022) | 1.2295 (+0.018) |
| INT6 Roundtrip BPB | 1.2444 | 1.2338 | 1.2385 |
| INT6 Sliding BPB | **1.2070** (−0.037) | **1.2107** (−0.023) | 1.2533 (+0.015) |
| Quant gap (roundtrip) | +0.014 | +0.017 | +0.027 |
| Steps | 153,381 | 84,278 | 161,364 |
| ms/step | 93.1 | 169.4 | 88.5 |
| Artifact size | 15.8 MB | 14.6 MB | 14.8 MB |

> **The `byte260` H-Net is closing the gap to `sp1024`.** At 4 hours, `byte260` reaches 1.2070 vs `sp1024`'s 1.2107 (INT6 sliding), effectively closing the gap with `sp1024` and the baseline (1.2074). Note that these are single runs with shared hyperparameters rather than individually tuned configurations. The key takeaway is that byte-level H-Net can match subword-level performance given sufficient compute.

Some remarks: After pre-QAT, two post-training effects impact the val_bpb:

1. **QAT needs per-model tuning.** We use the same QAT settings for both models (85% threshold, 25k warmdown, ~21k QAT steps). Both spike at QAT onset and only partially recover. Per-model tuning would likely help, but due to compute constraints, we haven't explored this further.

2. **`sp1024` chunk truncation hurts sliding window eval.**

   **The problem:** In the 4-hour table above, sliding window eval helps `byte260` (INT6 sliding 1.2070 vs roundtrip 1.2444, Δ=−0.037) but *hurts* `sp1024` div=4 (INT6 sliding 1.2533 vs roundtrip 1.2385, Δ=+0.015). The question is why?

   **Hypothesis:** With `CHUNK_DIVISOR=4`, boundaries are capped at `seq_len // 4 = 512`. The `sp1024` router predicts ~525 boundaries per sequence - 13 over the cap (based on our collected statistics). These excess boundaries sit at the tail and are effectively silently dropped. Sliding eval (stride=64) only scores the last 64 positions, which is exactly where truncation occurs. Roundtrip eval scores all 2048 positions, so the ~13 'impacted' positions are <1% of the total and barely affect the result.

   We confirmed this hypothesis with boundary stability measurements across 2000 windows, using `CHUNK_DIVISOR=2` and `CHUNK_DIVISOR=4`:

   | | `byte260` (div=4) | `sp1024` (div=2) | `sp1024` (div=4) |
   |--|------:|------:|------:|
   | Raw boundaries / seq | 461 | 465 | 525 |
   | Cap | 512 | 1024 | 512 |
   | **Truncated** | **No** | **No** | **Yes** |
   | Boundary flips per window shift | 1.7 | 1.8 | 2.9 |
   | Float→INT6 boundary flips | 2.6 | 2.5 | 5.4 |

   *Boundary flips =* how many positions change their boundary decision (boundary vs non-boundary) between two overlapping windows (window shift) or between the float and INT6 model on the same input (quant). Some flips are expected since shifting the window changes context, but fewer flips indicate more stable boundaries.

   > Another result that confirms this hypothesis is the switch to dynamic boundaries during inference (see results below).

   **Fix:** With `CHUNK_DIVISOR=2` (cap=1024), `sp1024` is no longer truncated (465 < 1024). Boundary stability matches `byte260` (1.8 vs 1.7 flips), and more importantly, sliding window now helps instead of hurting (INT6 sliding 1.2107 vs roundtrip 1.2338, Δ=−0.023), and the quantization gap drops from +0.027 to +0.017. However, using `CHUNK_DIVISOR=2` will result in wasted compute later in training since the cap is much higher than the average number of boundaries (1024 vs ~465).


### Dynamic chunking at inference

During training, we use a fixed boundary cap (`max_chunks = L // CHUNK_DIVISOR`), as explained above.

> **Does switching to a dynamic cap (set per-batch to the actual boundary count) at inference improve results, or has the router adapted to the fixed cap during training?**

To get a sense of this, we evaluate the 4-hour models with dynamic chunking at inference, where `max_chunks` is set per-batch to the actual maximum boundary count, rather than the fixed `L // CHUNK_DIVISOR` used during training.

**Results** for 4 hours runs:

| | `byte260` (div=4) | `sp1024` (div=2) | `sp1024` (div=4) |
|--|------:|------:|------:|
| Float Sliding (fixed cap) | 1.1980 | 1.1940 | 1.2295 |
| Float Sliding (dynamic) | 1.1912 | 1.1940 | 1.1960 |
| Δ float | −0.007 | 0.000 | −0.034 |
| INT6 Sliding (fixed cap) | **1.2070** | 1.2107 | 1.2533 |
| INT6 Sliding (dynamic) | **1.2049** | 1.2107 | 1.2201 |
| Δ INT6 | −0.002 | 0.000 | −0.033 |

- Dynamic inference improves or matches all models. The biggest gain is `sp1024` div=4 (INT6 sliding: 1.2533 → 1.2201), which also supports the truncation analysis above: the router was predicting correct boundaries, they were just being silently dropped by the fixed cap.

- For `byte260`, the effect is negligible since it wasn't truncated or very rarely truncated. `sp1024` div=2 shows exactly `Δ=0.000`: since it was never truncated (465 < 1024 cap), there's nothing to recover. However, this comes at the cost of wasted compute during training on ~559 padded positions per sequence.

*Note* that `sp1024` div=4 with dynamic inference (1.2201) still trails `sp1024` div=2 (1.2107). Removing the cap at inference restores the dropped boundaries, but can't undo the training-time 'damage' from 161k steps of truncation. This shows that choosing `CHUNK_DIVISOR` carefully is very important.

## Reproduction

```bash
# Best `byte260` 4-hour run (1.2070 BPB)
COMPILE_MODEL=1 OUTER_LAYERS=3 NUM_LAYERS=12 MODEL_DIM=512 \
    TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=14400 ITERATIONS=500000 \
    RATIO_LOSS_WEIGHT=0.05 LATE_QAT_THRESHOLD=0.85 WARMDOWN_ITERS=25000 \
    EMA_DECAY=0 EVAL_STRIDE=64 EVAL_BATCH_SEQS=16 \
    GPTQ_RESERVE_SECONDS=120 GPTQ_CALIB_BATCHES=512 PRUNE_PCT=0.03 \
    CHUNK_DIVISOR=4 CHECKPOINT_EVERY=50000 \
    MODEL_SAVE_PATH=models/byte260_compile_12L_OL3_4hr_qat85.pt \
    RUN_ID=byte260_compile_12L_OL3_4hr_qat85 \
    torchrun --standalone --nproc_per_node=8 hnet/train_gpt_hnet_compile_gptq.py

# 10 min equivalent
COMPILE_MODEL=1 OUTER_LAYERS=3 NUM_LAYERS=12 MODEL_DIM=512 \
    TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=600 ITERATIONS=20000 \
    RATIO_LOSS_WEIGHT=0.05 LATE_QAT_THRESHOLD=0.15 EMA_DECAY=0 \
    EVAL_STRIDE=64 EVAL_BATCH_SEQS=16 \
    GPTQ_RESERVE_SECONDS=45 GPTQ_CALIB_BATCHES=256 PRUNE_PCT=0.03 \
    CHUNK_DIVISOR=4 \
    MODEL_SAVE_PATH=models/sweep_div4_compile_10min.pt \
    RUN_ID=sweep_div4_compile_10min \
    torchrun --standalone --nproc_per_node=8 hnet/train_gpt_hnet_compile_gptq.py
```

> *Note*: All experiments use `seed=1337`

## Compliance

- [x] Artifact ≤16,000,000 bytes
- [x] 8×H100 training
- [x] No training on validation data
- [x] No network calls during evaluation
- [x] Non-record: extended run exceeds 10 min wallclock (**153k steps / 4h**)


## Credits

- **Paper**: Hwang et al. (2025), [*Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*](https://arxiv.org/abs/2507.07955) - the H-Net architecture this submission implements. Official code: [github.com/goombalab/hnet](https://github.com/goombalab/hnet)
- PRs including QAT / GPTQ / INT6 quantization / sliding window eval.
