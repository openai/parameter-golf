# N-gram Cache with Entropy-Adaptive Alpha

**val_bpb: 1.0945** (3-seed mean) | **~15.99 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.8.0+cu128)

| Seed | step_avg | steps | Sliding BPB | **N-gram BPB** | N-gram gain | N-gram time | Artifact |
|------|----------|-------|-------------|---------------|-------------|-------------|----------|
| 1337 | 97.7ms | 6,145 | 1.1263 | **1.0944** | -0.0319 | 66s | 15,863,727 |
| 42 | 97.7ms | 6,145 | 1.1268 | **1.0946** | -0.0322 | 64s | 15,988,183 |
| 2025 | 97.4ms | 6,164 | 1.1260 | **1.0945** | -0.0315 | 64s | 15,974,247 |
| **Mean** | **97.6ms** | **6,151** | **1.1264** | **1.0945 (std 0.0001)** | **-0.0319** | **~65s** | |

## Key Innovation: N-gram Cache with Entropy-Adaptive Alpha

Replaces TTT (test-time training) with a simple but effective N-gram cache that provides ~20x more BPB improvement:

- **N-gram cache** (order 7): During eval, builds a rolling cache of byte N-gram statistics from already-scored tokens
- **Entropy-adaptive alpha**: Instead of a fixed interpolation weight, scales alpha by per-token model uncertainty:
  ```
  effective_alpha = alpha * clamp(nll / threshold, 0.1, 2.0)
  ```
  When the model is confident (low NLL), the cache contribution is reduced. When uncertain (high NLL), the cache contributes more.
- **Strict backoff**: Falls back through N-gram orders (7 -> 6 -> ... -> 2) when context not found
- **CPU-only, overlapped**: N-gram scoring runs on a background CPU thread, overlapping with GPU sliding window eval

### Why N-gram > TTT

| Method | BPB gain | Eval time | Complexity |
|--------|----------|-----------|------------|
| TTT (3ep SGD) | -0.0025 | ~410s | High (gradient computation) |
| **N-gram cache** | **-0.0320** | **~65s** | Low (hash table lookups) |

TTT adapts model weights via gradient descent on already-scored tokens. N-gram cache instead directly interpolates simple count-based predictions with model logits — no gradients, no weight updates, just byte-level statistics. The improvement is 12x larger and 6x faster.

## Training Architecture

PR #414 stack:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Muon (WD=0.04) + Adam (WD=0.04) |

### N-gram Cache Hyperparameters

| Parameter | Value |
|-----------|-------|
| Max order | 7 |
| Alpha | 0.50 |
| NLL threshold | 2.5 |
| Adaptive range | clamp(nll/threshold, 0.1, 2.0) |
| Backoff | Strict (7 -> 6 -> ... -> 2) |

### Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (<=10 min) |
| Sliding window eval (8 GPU, stride 64) | ~107s |
| N-gram cache (CPU, overlapped) | ~65s |
| **Total eval** | **~107s (< 10 min)** |

N-gram runs concurrently with sliding window on CPU, so total eval time is max(sliding_window, ngram) rather than the sum.

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
POLYGLU_ENABLED=0 \
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=7 NGRAM_ALPHA=0.50 \
NGRAM_NLL_THRESHOLD=2.5 \
SEED=1337 RUN_ID=ngram_seed1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation

Incremental contribution (seed 1337):

| Configuration | BPB | Delta |
|---------------|-----|-------|
| Base model (int6 roundtrip) | 1.1498 | -- |
| + Sliding window (stride 64) | 1.1263 | -0.0235 |
| + N-gram cache (fixed alpha=0.5) | 1.0944 | -0.0319 |
| **vs SOTA (TTT-based, 1.1194)** | **1.0944** | **-0.0250** |

## Credits

- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **LeakyReLU^2 activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **EMA + SWA**: [PR #414](https://github.com/openai/parameter-golf/pull/414) stack
- **N-gram cache**: Novel addition — entropy-adaptive interpolation of byte-level N-gram statistics with neural LM logits
