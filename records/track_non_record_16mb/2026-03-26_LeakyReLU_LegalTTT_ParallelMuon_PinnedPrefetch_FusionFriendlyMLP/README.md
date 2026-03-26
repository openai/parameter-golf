# LeakyReLU^2 + Legal TTT + Parallel Muon + systems: prefetch & fusion-friendly MLP

Reference baseline: [`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`](https://github.com/openai/parameter-golf/pull/549)

## Outcome

This variant improves throughput slightly, but does **not** improve quality versus the original 3-seed 8xH100 runs.

- Mean steps in 600s: **7184.7 -> 7191.3** (+6.7 steps)
- Mean `step_avg`: **83.53ms -> 83.44ms** (faster)
- Mean pre-TTT `val_bpb` (`final_int6_sliding_window_exact`): **1.12184 -> 1.12334** (worse by +0.00151)
- Mean post-TTT `val_bpb` (`legal_ttt_exact`): **1.11938 -> 1.12096** (worse by +0.00158)

## 3-seed comparison (8xH100, 600s train budget)

| Seed | Baseline steps / post-TTT bpb | This run steps / post-TTT bpb | Delta |
|------|-------------------------------|--------------------------------|-------|
| 42 | 7182 / 1.12002032 | 7189 / 1.12119101 | +7 steps, +0.00117069 bpb |
| 1337 | 7179 / 1.11922988 | 7191 / 1.12088391 | +12 steps, +0.00165403 bpb |
| 2025 | 7193 / 1.11888882 | 7194 / 1.12081146 | +1 step, +0.00192264 bpb |
| **Mean** | **7184.7 / 1.11937967** | **7191.3 / 1.12096213** | **+6.7 steps, +0.00158245 bpb** |

## 1xH100 ablation (Modal sanity check, 600s train budget)

| Configuration | Steps / ms per step | Post-TTT bpb | Delta vs base |
|---------------|---------------------|--------------|---------------|
| Base record train_gpt | 924 / 649.71ms | 1.55027402 | - |
| + prefetch only | 942 / 637.55ms | 1.53744178 | +18 steps, -0.01283224 bpb |
| + prefetch + MLP fusion form | 943 / 636.73ms | 1.53642888 | +19 steps, -0.01384514 bpb |

## Interpretation

The data is consistent across all three seeds: the systems changes increase training throughput, but that throughput gain does not translate into better final validation quality in this setup.

So the result here is best described as a **speed optimization with neutral-to-slightly-negative quality impact** relative to the original record recipe. Likely just means **noise impacted the training result**, as training math and process is exactly the same.

On 1xH100, the same systems changes looked clearly positive (more steps and better post-TTT bpb), while on 8xH100 they remain speed-positive but quality-negative. The practical interpretation is that prefetch/fusion behavior does not transfer linearly from single-GPU to multi-GPU quality outcomes and should be treated as a throughput optimization first. **Likely, I/O is no longer bottleneck at large scale, and more so communication between GPUs tend to be the target**.

I will continue iterating on this as increased training speed shows promises. This attempt tries to prove that **async prefetching and memory pinning can improve the throughput of most approaches**, but requires more experimentation to investigate compatibility with other methods. Aiming to **increase optimization's compatibility with parallel GPUs next**.

# What changed vs. base record

All differences are in **data loading** and **MLP forward**; model architecture, banking, Parallel Muon, FlashAttention-3, `torch.compile` usage, TTT protocol, and env-driven hyperparameters are otherwise aligned with base PR

### 1. Pinned async prefetch (`PrefetchingDistributedTokenLoader`)

- **Imports:** `queue`, `threading`.
- **Hyperparameters (env):**
  - `TRAIN_PREFETCH` (default `1`)
  - `TRAIN_PREFETCH_QUEUE` (default `2`)
  - `TRAIN_COPY_STREAM` (default `1`) — when enabled with prefetch, H2D uses a dedicated `torch.cuda.Stream` and the default stream waits on it.
- **Helpers:** `_cpu_batch_from_stream`, `_h2d_int64_batches`.
- **Loader:** daemon thread builds the next `(x, y)` on CPU, `contiguous().pin_memory()`, bounded `queue.Queue`; `next_batch` dequees and copies to device.
- **Training loop:** `make_train_loader()` factory; after **optimizer state rewind** (e.g. SWA branch), existing prefetch thread is **`shutdown()`** before a fresh loader is created so the token stream does not advance in the background.

### 2. Fusion-friendly LeakyReLU² MLP

**Base:**

```python
x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
return F.linear(x.square(), down_w.to(x.dtype))
```

**This submission:**

```python
x_dtype = x.dtype
up_w = up_w.to(dtype=x_dtype)
down_w = down_w.to(dtype=x_dtype)
h = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5)
return F.linear(h * h, down_w)
```

Mathematically identical to LeakyReLU(0.5)² feeding the down projection; the change is **layout / fusion hints** for the compiled training graph, the Inductor fuses or simplifies more than before.

## ENV

Same as the base run command, with optional prefetch toggles (defaults match optimized script):

```bash
TRAIN_PREFETCH=1 TRAIN_PREFETCH_QUEUE=2 TRAIN_COPY_STREAM=1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base model and training recipe:** PR #549
- **Fusion / step-time motivation:** PR #640
- **Async prefetching:** PR #591