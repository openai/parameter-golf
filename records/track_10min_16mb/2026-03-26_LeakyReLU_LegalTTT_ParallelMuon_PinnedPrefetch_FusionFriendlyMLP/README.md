# LeakyReLU² + Legal TTT + Parallel Muon — **systems: pinned async prefetch + compiler fusion-friendly MLP**

**val_bpb: ~1% less after simple improvement** 
**Steps: 2% more steps in 600s**

This submission is the [2026-03-23 LeakyReLU + Legal TTT + Parallel Muon](https://github.com/openai/parameter-golf/pull/549) training stack with **two additional code paths** aimed at **more training steps in 600s**:

1. **Pinned async training batch prefetch** — background CPU work (token slice → reshape → `pin_memory`) overlapped with GPU compute; host-to-device copies on an optional dedicated CUDA stream so `non_blocking=True` transfers can overlap.
2. **Compiler fusion-friendly LeakyReLU² MLP** — same math as the base (`leaky_relu(·, 0.5)` then square into the down projection), rewritten as `h * h` with explicit weight casting so `torch.compile(fullgraph=True)` can better fuse elementwise work and avoid an extra temporary. Taken from [73.7M Ternary U-Net + NeoMuon + 4x relu²MLP + Factored Tied Emb + Poly5 Softcap + YaRN2048 + 8192BPE + FP8QAT + Bitmask-LZMA + Stride-16 Sliding](https://github.com/openai/parameter-golf/pull/640)

Mainly, this submission proves that **simple application of async prefetching and memory pinning can elevate most approaches** by slightly noticeable amounts.

---

## Results (8×H100)

TBD

---

## What changed vs. base record

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

---

## Iteration log

I set out to make improvements to current records by applying my intuitive optimisation that I thought of in my [Nonrecord Async Prefetching PR](https://github.com/openai/parameter-golf/pull/591). Though Triton, cuBLAS and FA3 are already known to minimize graph breaks and idle time, I wanted to verify if my approach can still make further optimizations. On multiple GPUs with high computing power, the scale at which async prefetching can improve the results increases.

### A. Baseline

- Ran original record script on Modal 1xH100 GPU.
- 924 steps in 600s, val_bpb = 1.55027402 after ttt

### B. + Async prefetching

- Ran script applying async prefetching (pinned CPU batches, optional copy stream, queue depth).
- 942 steps in 600s, val_bpb = 1.53744178 after ttt
- Observed ~**20 extra training steps** vs. baseline under the same compute.

### C. + Prefetch + fusion-friendly MLP

- Ran script applying the fusion-friendly matmul.
- 943 steps in 600s, val_bpb = 1.53642888 after ttt
- Observed ~**20 extra training steps** vs. baseline 
- Results too similar to Async prefetching only, unable to verify if effective yet.

### D. Scale-out verification

- 8×H100 (Runpod): In progress

---

## Ablation table

The above results in table form. Will update with more information after Runpod test.

| Configuration | Steps / ms per step (600s wall clock) | Post-TTT bpb |
|---------------|----------------------------------------|--------------|
| Base record train_gpt | 924 / 649.71 | 1.55027402 |
| + prefetch only | 942 / 637.55 | 1.53744178 |
| + prefetch + MLP fusion form | 943 / 636.73 | 1.53642888 |

---

## Credits

- **Base model and training recipe:** [PR #549 LeakyReLU + Legal TTT + Parallel Muon](https://github.com/openai/parameter-golf/pull/549) (PR #414 / #399 / #461 / #493 / #518 as cited there).
- **Fusion / step-time motivation:** [PR #640 Ternary U-Net record RESULTS.md](https://github.com/openai/parameter-golf/pull/640) (compiler fusion-friendly MLP)
- **Async prefetching:** [PR #591](https://github.com/openai/parameter-golf/pull/591)