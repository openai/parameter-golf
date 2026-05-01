# Non-Record V5: 7-Layer UNet + Int8 QAT + EMA + 4 h Training + Sliding-Window + N-gram Eval

**val_bpb: 1.219329** (single seed 42, stride-64 sliding window + n-gram cache) | **vs V4: -0.09019312 BPB** | DGX Spark (1× GB10 Blackwell)

> *Final BPB, artifact size, and delta vs V4 will be filled in once the post-training eval completes.*

## 1. Overview

V5 continues the [V4 submission](../2026-04-23_SharedTransformer_Int6_SlidingWindow_7L) lineage. The primary changes vs V4 are:

1. **Quantization**: reverted from int6 per-row + LZMA to **int8 per-row + zlib** (`USE_INT6=0`). This uses the simpler and slightly faster quantization path; int6+lzma gave a smaller artifact (13.7 MB) but int8+zlib still comfortably fits under 16 MB and avoids multi-percentile sweep overhead at compression time.
2. **N-gram cache mixture**: post-training, we mix neural token probabilities with a token-level n-gram backoff cache (`eval_spark.py`). Alpha and order are selected by a fast sweep over 2M tokens; the best combo is applied to the full validation set. This methodology was developed for V3+.
3. **Eval pipeline**: sliding window (stride=64) + n-gram cache mixture, same as V4. This consistently improves BPB by ~0.05–0.08 over chunked eval.

Architecture, optimizer LRs, EMA, and QAT schedule are identical to V4.

## 2. Architecture

| Property | Value |
|---|---|
| Transformer layers | 7 |
| Topology | U-Net skip connections (encoder/decoder halves, learned skip weights) |
| Model dim | 512 |
| Attention heads | 8 query, 4 KV (GQA, head_dim 64) |
| MLP multiplier | 4× |
| MLP activation | Leaky ReLU squared (negative_slope=0.5, then squared) |
| Embeddings | Tied, separate `tied_embed_lr` |
| Logit softcap | Tanh, softcap=30.0 |
| Positional encoding | RoPE (base 10000.0) |
| Vocab | SentencePiece BPE, 1024 tokens |
| Sequence length | 1024 |
| Total params | **20,725,304** |

## 3. Training Hyperparameters

| Hparam | Value | Same as V4? |
|---|---|---|
| `MATRIX_LR` | 0.08261619767374824 | yes |
| `SCALAR_LR` | 0.014691154447587356 | yes |
| `TIED_EMBED_LR` | 0.021552090970329115 | yes |
| `HEAD_LR` | 0.0 | yes |
| `MUON_MOMENTUM` | 0.9382982028913158 | yes |
| `WARMDOWN_ITERS` | 1558 | yes |
| `EMA_DECAY` | 0.997 | yes |
| `QAT_ENABLED` | 1 | yes |
| `USE_INT6` | **0** | **changed (V4 used 1)** |
| `MAX_WALLCLOCK_SECONDS` | 14400 (4 h) | **changed (V4 used 36000 = 10 h)** |
| `TRAIN_BATCH_TOKENS` | 524,288 | yes |
| `TRAIN_SEQ_LEN` | 1024 | yes |
| `SEED` | 42 | yes |

LRs were discovered by Optuna search; same config as V3.1 and V4.

## 4. Multi-GPU Compatibility

The training script (`train_gpt.py` in this record) has full DDP support:

- Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from environment
- Wraps model in `DistributedDataParallel` when distributed
- Scales `grad_accum_steps = 8 // world_size` (8 grad-accum on 1 GPU → 1 on 8×H100)
- Muon optimizer uses `dist.all_reduce` for cross-rank gradient sync
- Wall-clock cap synced across ranks via `dist.all_reduce`

Run on 8×H100 with:
```bash
NCCL_IB_DISABLE=1 NUM_LAYERS=7 MLP_MULT=4 MATRIX_LR=0.08261619767374824 \
SCALAR_LR=0.014691154447587356 TIED_EMBED_LR=0.021552090970329115 \
MUON_MOMENTUM=0.9382982028913158 WARMDOWN_ITERS=1558 \
EMA_ENABLED=1 EMA_DECAY=0.997 QAT_ENABLED=1 USE_INT6=0 \
MAX_WALLCLOCK_SECONDS=600 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Run locally on DGX Spark (1× GB10):
```bash
NUM_LAYERS=7 MLP_MULT=4 MATRIX_LR=0.08261619767374824 \
SCALAR_LR=0.014691154447587356 TIED_EMBED_LR=0.021552090970329115 \
MUON_MOMENTUM=0.9382982028913158 WARMDOWN_ITERS=1558 \
EMA_ENABLED=1 EMA_DECAY=0.997 QAT_ENABLED=1 USE_INT6=0 \
MAX_WALLCLOCK_SECONDS=14400 SEED=42 \
CHECKPOINT_FILENAME=final_model_seed42.int8.ptz \
python3 train_gpt.py
```

## 5. Post-Training Evaluation

After training, we run the eval pipeline in `eval_spark.py`:

```bash
# Chunked + n-gram baseline
python3 eval_spark.py --checkpoint final_model_seed42.int8.ptz --alpha 0.2 --max-order 5

# Sliding window (stride=64) + n-gram
python3 eval_spark.py --checkpoint final_model_seed42.int8.ptz --stride 64 --alpha 0.2 --max-order 5

# Alpha/order sweep (2M tokens), then full-val run at best combo
python3 eval_spark.py --checkpoint final_model_seed42.int8.ptz \
    --stride 64 --alpha 0.2 --max-order 5 \
    --sweep --sweep-tokens 2000000
```

`ngram_cache.py` contains the `TokenNGramCache` class used by `eval_spark.py`.

## 6. Quantization: Int8 per-row + zlib

Uses the standard `quantize_state_dict_int8` path in `train_gpt.py`:
- Per-row int8 for 2D float tensors (weights)
- Per-tensor int8 for vectors/scalars
- Small tensors (≤65536 elements) kept as fp16 passthrough
- Final blob compressed with `zlib.compress(level=9)`

## 7. Results

*(To be filled in after training and eval complete.)*

| Metric | Value |
|---|---|
| Training steps | TBD |
| Step avg (ms) | TBD |
| Pre-quant BPB (chunked) | TBD |
| Post-quant BPB (chunked + n-gram α=0.2, ord=5) | TBD |
| **Sliding BPB (stride=64 + best n-gram)** | **TBD** |
| Artifact bytes | TBD |
| Total submission bytes | TBD |

## 8. Lineage

```
V2 (1.39693 BPB, int8 QAT, 4h, chunked) — first int8 QAT result
  └── V3.1 (1.36409 BPB, int8 QAT, 4h, sliding window stride=64) — added sliding eval
        └── V4 (1.30952 BPB, int6 QAT, 10h, sliding + n-gram) — int6+lzma + 10h training + n-gram
              └── V5 (TBD BPB, int8 QAT, 4h, sliding + n-gram) — reverted to int8+zlib, kept n-gram
```
