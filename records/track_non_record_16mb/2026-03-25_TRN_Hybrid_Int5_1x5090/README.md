# Temporal Resonance Network (TRN) Hybrid for Parameter Golf

## Central Question

Can a linear recurrence architecture compete with attention under extreme parameter
constraints (16 MB, 10 minutes)? TRN offers O(n log n) training via parallel scan
and frequency-domain compression of sequential patterns -- but its selective recall
is weak (8.8% on synthetic copy tasks vs 96.2% for attention). This entry tests a hybrid:
7 TRN layers for pattern compression, 3 attention layers for exact retrieval,
quantized to int5 to fit 10 layers within the 16 MB budget.

## Score

val_bpb = 1.4942 (single seed, int5+zstd-22 roundtrip, 636 steps / 600s wallclock on single RTX 5090)
Artifact size: 15.28 MB (int5+zstd-22, code+model)

Note: The 636-step score (wallclock-capped at 600s) is the best int5 roundtrip we
achieved. At 20K steps the fp32 model reaches 1.26 bpb, but int5 quantization
degrades it to 1.93 bpb (+0.67). The root cause is analyzed in the Quantization
section below. This is a non-record submission focused on the architecture and
the lessons learned.

---

## Architecture

10 layers: 7 TRN + 3 causal attention, interleaved.

```
[TRN][TRN][Attn][TRN][TRN][Attn][TRN][TRN][Attn][TRN]
  0    1    2     3    4    5     6    7    8     9
```

- d_model = 512, K = 256 oscillators per TRN layer, vocab = 1024
- SwiGLU FFN (mlp_mult=2), U-Net skip connections, tied embeddings
- Position-aware oscillators replace standard positional encoding --
  omega * log(1+t) is a deterministic phase offset (functionally a form of PE),
  but with learned per-oscillator frequencies fused into the recurrence
- BigramHash(10240, dim=128): token-pair hash table added to embedding

### Parameter Breakdown

| Component | Params | % |
|-----------|--------|---|
| TRN oscillator projections (7L, d_model->6K) | 5.6M | 22% |
| TRN W_res (7L, 2K->d_model) | 1.8M | 7% |
| Attention QKV+out (3L, GQA 8/4 heads) | 2.4M | 9% |
| FFN SwiGLU (10L, 2x expansion) | 10.5M | 41% |
| BigramHash (10240 x 128 + projection) | 1.8M | 7% |
| Embeddings (1024 x 512, tied) | 0.5M | 2% |
| Other (norms, biases, skip weights) | 3.2M | 12% |
| **Total** | **25.8M** | **100%** |

### TRN Recurrence

Each TRN layer maintains K complex-valued oscillators:

```
drive_t = A_t * exp(j * (omega * log(1 + t) + phi_t))
r_t     = alpha_t * r_{t-1} + (1 - alpha_t) * drive_t
y_t     = Re(r_t * exp(-j * omega * log(1 + t)))
```

A (amplitude), omega (frequency), phi (phase), alpha (decay gate) are projected
from each input token via a single linear layer (d_model -> 6K). alpha is
sigmoid-gated with multi-scale initialization (half-life ~1.4 / ~2.9 / ~33 steps).

The log-time warp (log(1+t)) prevents frequency aliasing at long range.
Parallel training uses a Kogge-Stone prefix scan -- O(n log n), pure PyTorch.

Related work: S4, S4D/DSS, LRU, Mamba, RWKV. Our variant combines complex
oscillatory state with learned per-oscillator frequency/phase and log-time
warping -- distinct from LRU's complex diagonal recurrence.

### Why Hybrid

TRN alone cannot reliably recall discrete tokens; attention alone cannot fit
competitive depth within the 16 MB budget at int8.

Attention handles exact content-addressed retrieval -- locating a specific token
at an arbitrary position. TRN compresses sequential patterns into frequency-domain
state, which is efficient for repetitive structure and positional correlations but
weak at selective discrete recall (8.8% vs 96.2% on synthetic copy tasks; FineWeb
equivalent not directly measurable). Interleaving attention layers injects
exact-retrieval capacity periodically, preventing gradient starvation in deep
TRN stacks.

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Layers | 10 (7 TRN + 3 Attn) |
| d_model | 512 |
| Oscillators K | 256 |
| Vocab | 1024 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP mult | 2 (SwiGLU) |
| Seq len | 1024 |
| Batch tokens | 262144 |
| Iterations | 20000 |
| Warmup steps | 20 (torch.compile cache warmup, not LR warmup) |
| Warmdown iters | 1200 |
| Muon (matrices) | lr=0.04, momentum=0.95 |
| Adam (embeddings) | lr=0.05, beta=(0.9, 0.95) |
| Grad clip norm | 0.3 |
| Token shift | enabled (RWKV-6 style pre-resonance mixing) |
| Activation | LeakyReLU(0.5)^2 |
| PCG lambda | 0.5 |
| Weight decay | 0.04 (Muon matrices only) |
| EMA | decay=0.997, start at 50% of training |
| Compression | zstd level 22 |

---

## Quantization

### The int5 trade at 1000 steps

**Depth-for-quantization trade:** int5 (vs int8) frees ~4 MB, allowing 10 layers
instead of 7 within the 16 MB budget. 10L int8 is 18.6 MB -- over the limit.

Int5 per-row symmetric: weights mapped to [-15, 15], scale stored as fp32.
Applied to all matrix weights; embeddings remain fp16.
5 bits/weight, 0.625 bytes/weight.
Size: 25.8M * 0.625 = 16.1 MB raw + scales (~0.4 MB) + fp16 embeddings (~1 MB).
zstd-22 compresses the blob to 15.28 MB.

QAT uses straight-through estimator (STE): forward pass quantizes weight.data,
backward pass updates the fp32 copy.

| Quantization | bpb (10L, 1000 steps) | Degradation vs fp32 | Artifact |
|--------------|-----------------------|---------------------|----------|
| fp32 (train) | 1.4537 | -- | 103 MB |
| int8 + zstd-22 | 1.4554 | +0.002 | 18.6 MB (over limit) |
| int5 + zstd-22 (no QAT) | 1.5254 | +0.072 | 15.27 MB |
| int5 + zstd-22 (QAT frac=0.3) | 1.4942 | +0.041 | 15.28 MB |

QAT reduces int5 degradation from +0.072 to +0.041 (a 0.031 bpb improvement).
The 7L int5+QAT model (1.4963) and 10L int5+QAT model (1.4942) achieve
similar bpb despite the depth difference -- at 1000 steps, the extra layers
have not yet contributed. Longer training is needed for the depth trade to pay off.

### The int5 collapse at 20K steps

At 20K steps, int5 shows severe degradation:

| Quantization | bpb (10L, 20K steps) | Degradation |
|--------------|----------------------|-------------|
| fp32 | 1.2633 | -- |
| int8 + zstd-22 | 1.2711 | +0.008 |
| int5 + zstd-22 | 1.9321 | +0.669 |

The int5 degradation grows from +0.041 at 1000 steps to +0.669 at 20K steps.
Int8 remains healthy (+0.008), ruling out a general quantization problem.

**Root cause: TRN oscillator projection weights are structurally incompatible
with 5-bit quantization at convergence.**

The oscillator projection (d_model -> 6K) encodes omega (frequency), phi (phase),
alpha (decay), and amplitude for each oscillator. These parameters control
`sin(omega * t + phi)` -- a 1-bit error in omega accumulates as O(t) phase drift
across the sequence. At 1000 steps, weight magnitudes are small and quantization
bins are close together, so errors stay bounded. At 20K steps, the distribution
sharpens: oscillator projections develop precise frequency-selective patterns
where small perturbations cascade through the recurrence.

Supporting evidence:
- A parameter-matched 13L Transformer (25.8M params, same codebase) shows only
  +0.016 bpb int5 degradation at 1000 steps -- Transformer weights do not encode
  phase-sensitive recurrence parameters.
- The TRN hybrid's int8 artifact (20.1 MB) exceeds 16 MB, while the 13L
  Transformer's int8 artifact (15.7 MB) fits. TRN oscillator weights have higher
  entropy, resisting zstd compression (1.30x vs 1.66x for Transformer).
- QAT with 6000 STE steps did not recover the int5 quality, suggesting the
  quantization error surface for oscillatory parameters is not smoothly navigable
  by gradient descent through STE.

This is an open problem. Possible directions include mixed-precision quantization
(int8 for oscillator projections, int4 for FFN), learned quantization grids
per component, or architectural changes to reduce phase sensitivity (e.g.,
discretizing omega to a fixed grid before training).

---

## Internal Ablation: TRN Hybrid vs Transformer

### Short training (1000 steps, 600s wallclock, single RTX 5090)

Both models trained within the same codebase (train_gpt_trn.py), same optimizer,
same BigramHash(10240), grad_clip=0.3,
int5 QAT, zstd-22.

**Important caveats:**
- This is an internal ablation within one codebase, not a comparison against
  the fully-optimized leaderboard baseline (train_gpt.py, 1.14-1.22 bpb).
- TRN hybrid has 27% more parameters (25.8M vs 20.3M). The bpb advantage is
  partially attributable to this difference.
- Single run, no variance bars. Results are indicative, not statistically conclusive.
- Transformer step_avg includes a QAT compile spike at step 2 (49.8s); steady-state
  is ~1443 ms/step. TRN steady-state is ~945 ms/step (1.53x faster).

| Metric | Transformer 10L | TRN Hybrid 10L | Delta |
|--------|----------------|----------------|-------|
| Batch tokens/step | 262144 | 262144 | same |
| ms/step (steady-state) | ~1443 | ~945 | -35% |
| Steps in 600s | 379 | 636 | +68% |
| Total tokens seen | 199M | 334M | +68% |
| val_bpb (wallclock stop) | 1.5983 | 1.4715 | -0.127 |
| val_bpb (int8 roundtrip) | 1.6252 | 1.4755 | -0.150 |
| Params | 20.3M | 25.8M | +27% |
| Int5 artifact | 11.9 MB | 15.3 MB | |

At the same batch size, TRN hybrid is 35% faster per step (steady-state) and sees
68% more tokens in the same 600-second budget. It achieves 0.127 bpb lower loss.

### Parameter-matched comparison (1000 steps)

A 13-layer Transformer (25,785,449 params) was trained under identical conditions
to provide a parameter-matched comparison.

| Metric | Transformer 13L | TRN Hybrid 10L |
|--------|----------------|----------------|
| Params | 25.79M | 25.79M |
| ms/step (steady-state) | ~1131 | ~945 |
| fp32 val_bpb | 1.3527 | 1.4537 |
| int5 val_bpb | 1.3689 (+0.016) | 1.4942 (+0.041) |
| int5 artifact | 15.30 MB | 15.28 MB |
| VRAM | 21,574 MiB | ~19,500 MiB |

At 1000 steps, the 13L Transformer achieves lower bpb than the 10L TRN hybrid
(-0.101). The Transformer's advantage here likely reflects two factors:
(1) 13 layers of attention vs 10 layers (3 attention + 7 TRN) provides more
capacity for the content-retrieval tasks dominant in early training, and
(2) the TRN's Kogge-Stone scan, while asymptotically efficient, has not yet
converged its oscillator parameters at 1000 steps.

The TRN hybrid's per-step speed advantage (16% faster) partially offsets this,
yielding more steps per wallclock budget. Whether this advantage grows with
longer training is shown in the convergence section below.

### Long training (20K steps, single RTX 5090)

| Metric | TRN Hybrid 10L (20K) |
|--------|---------------------|
| fp32 val_bpb | 1.2633 |
| int8 val_bpb | 1.2711 (+0.008) |
| int5 val_bpb | 1.9321 (+0.669) |
| int8 artifact | 20.1 MB (over limit; 18.6 MB at 1000 steps -- weights compress less as they specialize) |
| int5 artifact | 15.4 MB |
| step_avg | ~870 ms |

The fp32 model continues to improve (1.45 at 1K -> 1.26 at 20K), and int8
faithfully tracks it. But int5 collapses, making the 20K model unusable under
the 16 MB constraint with uniform int5. This is the central failure of this
submission.

---

## Convergence (10L, single RTX 5090, 20K steps)

| Step | fp32 val_bpb | int8 val_bpb | int5 val_bpb |
|------|-------------|-------------|-------------|
| 636 | 1.4715 | 1.4755 | ~1.49 (QAT) |
| 2000 | 1.3874 | -- | -- |
| 6000 | 1.3307 | -- | -- |
| 10000 | 1.3167 | -- | -- |
| 14000 | 1.3100 | -- | -- |
| 18000 | 1.3044 | -- | -- |
| 20000 | 1.2633 | 1.2711 | 1.9321 |

The fp32 trajectory shows steady improvement through 20K steps with no sign
of saturation. The gap between fp32 and int5 widens monotonically with training,
confirming that the quantization problem worsens as oscillator weights specialize.

---

## Ablation: Layout (7 layers, 1000 steps)

| Layout | attn_positions | val_bpb (int8) |
|--------|----------------|----------------|
| Front-loaded [AA TTTTT] | 0, 1 | 1.5411 |
| Interleaved [TT A TT A T] | 2, 5 | 1.5331 |

Interleaved diverges after ~500 steps -- periodic exact-retrieval beats front-loading.

---

## Ablation: Stacked Optimizations (7 layers, 1000 steps)

Each row adds to the previous. All measured on single RTX 5090.

| Config | int8 bpb |
|--------|----------|
| Interleaved baseline | 1.5331 |
| + BigramHash(10240) | 1.5303 |
| + grad_clip=0.3, token_shift, LeakyReLU^2, PCG=0.5 | 1.4963 |

The last four changes were applied together (-0.034 bpb combined).
Individual contributions are not isolated.

---

## Ablation: Depth + Quantization (1000 steps)

| Config | Params | int5 bpb | Artifact |
|--------|--------|----------|----------|
| 7L int8 (no QAT) | 18.7M | -- (14.5 MB) | fits |
| 7L int5 + QAT | 18.7M | 1.4963 | 10.82 MB |
| 10L int5 + QAT | 25.8M | 1.4942 | 15.28 MB |

The 10-layer int5 model matches the 7-layer int8 result at 1000 steps.

---

## FLOP Analysis: TRN vs Transformer Per Layer

| Component | Transformer | TRN Block |
|-----------|-------------|-----------|
| Projections (QKV / oscillator) | 0.40 GFLOP | 0.81 GFLOP |
| Attention / scan | 0.27 GFLOP | 0.002 GFLOP |
| Output proj (out / W_res) | 0.27 GFLOP | 0.54 GFLOP |
| FFN SwiGLU | 1.07 GFLOP | 1.21 GFLOP |
| Other (norms, skip, token_shift) | -- | ~0.12 GFLOP |
| **Total per layer** | **2.01 GFLOP** | **2.68 GFLOP** |

TRN's overhead is not in the scan (0.3% of total FLOPs) but in the oscillator
projection (d_model -> 6K) and W_res output mapping (2K -> d_model). The scan
itself is cheaper than attention, but at seq_len=1024 this difference is small.

Total model: 7 TRN + 3 Attn = 24.8 GFLOP vs 10 Attn = 20.1 GFLOP (1.23x).
Despite more FLOPs, TRN is faster per step because attention layers are
memory-bandwidth bound with synchronization points (softmax, causal mask),
while TRN's Kogge-Stone prefix scan in pure PyTorch streams with fewer
synchronization points. FLOPs alone do not predict wall-clock time for
memory-bound operations.

---

## What We Tried (That Didn't Work)

| Experiment | Result | Lesson |
|------------|--------|--------|
| Depth recurrence (TRN 2-pass weight reuse) | +0.06 bpb, 1.7x slower | omega*t phase is identical in both passes -- second pass recomputes the same function on refined input. Pass-dependent phase offset could help but was not attempted. |
| K=128 + MLP 3x expansion | +0.01 bpb vs K=256 + MLP 2x | Oscillator count matters more than FFN width for TRN |
| SWA/EMA (decay=0.997, 1000 steps) | +0.016 bpb | Warmdown too short at 1000 steps |
| TTT LoRA rank=16 | Same as rank=8 | 1 Adam step cannot move large LoRA; rank=8 is sufficient |
| EMA + QAT at 20K steps | int5 +0.669 bpb | EMA does not help int5; oscillator weights are the bottleneck |
| QAT with 6000 STE steps at 20K | int5 +0.669 bpb | STE cannot navigate the error surface of oscillatory parameters at convergence |

---

## Reproducibility

Single GPU (reproduces the submitted score -- runs 1000 iterations with 600s wallclock cap, stops at ~636 steps):

```bash
INT5_QAT=1 INT5_QAT_START_FRAC=0.3 \
USE_ZSTD=1 ZSTD_LEVEL=22 \
ITERATIONS=1000 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=50 \
MODEL_TYPE=hybrid NUM_LAYERS=10 ATTN_POSITIONS=2,5,8 \
BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 \
GRAD_CLIP_NORM=0.3 USE_TOKEN_SHIFT=1 \
TRAIN_BATCH_TOKENS=262144 \
SEED=42 \
python train_gpt_trn.py
```

8xH100 (untested -- intended for future leaderboard runs if quantization is resolved):

```bash
INT5_QAT=1 INT5_QAT_START_STEP=2000 \
USE_ZSTD=1 ZSTD_LEVEL=22 \
ITERATIONS=20000 \
MODEL_TYPE=hybrid NUM_LAYERS=10 ATTN_POSITIONS=2,5,8 \
BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 \
GRAD_CLIP_NORM=0.3 USE_TOKEN_SHIFT=1 \
WEIGHT_DECAY=0.04 \
TRAIN_BATCH_TOKENS=262144 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt_trn.py
```

The 20K run used TRAIN_BATCH_TOKENS=262144 explicitly via run_overnight.sh.

Note: at 20K steps the fp32 model reaches 1.26 bpb, but int5 collapses to
1.93 bpb. The submitted score uses the 1000-step single-GPU command above.

Unlisted env vars use code defaults: d_model=512, K=256, vocab=1024,
heads=8, kv_heads=4, mlp_mult=2, seq_len=1024, warmdown_iters=1200.

Requires: `pip install zstandard sentencepiece`

Data: `python3 data/cached_challenge_fineweb.py --variant sp1024`

---

## Implementation Notes

- Pure PyTorch -- no Triton, no custom CUDA extensions
- Kogge-Stone parallel prefix scan: log-depth tree reduction over complex recurrence
- torch.compile compatible (fullgraph=False; scan loop unrolls at fixed seq_len)
- Int5 bit packing in NumPy at save/load; inference uses fp32 dequantized weights
- All TRN modules inlined in train_gpt_trn.py -- zero external dependencies

---

## Known Weaknesses and Open Problems

1. **Int5 quantization collapse at convergence.** The central limitation. TRN oscillator
   projections encode frequency/phase parameters where quantization errors accumulate
   as O(t) phase drift. At 1000 steps the error is bounded (+0.041); at 20K steps
   it is catastrophic (+0.669). This makes uniform int5 unsuitable for converged
   TRN models under the 16 MB constraint. Possible mitigations (untested):
   mixed-precision quantization, learned quantization grids, or omega discretization.

2. **No int8 path within 16 MB.** TRN oscillator weights have high entropy,
   resisting zstd compression (1.30x vs 1.66x for Transformer at int8). The 10L
   hybrid int8 artifact is 20.1 MB -- 25% over the limit. Reducing model size
   to fit int8 would sacrifice the depth advantage that motivated int5.

3. **Parameter-matched comparison favors Transformer at 1000 steps.** The 13L
   Transformer reaches 1.3689 int5 bpb vs the hybrid's 1.4942 at the same param
   count and step count. TRN's per-step speed advantage (16%) partially compensates
   under wallclock constraints, but the gap is real.

4. **The four stacked optimizations (token_shift, LeakyReLU^2, PCG, grad_clip) are
   not individually ablated.** Some may be inactive.

5. **TRN's selective copy accuracy (8.8%) means the architecture depends on
   attention layers for discrete recall.** The 3/10 attention ratio is a floor.

---

## Ongoing Work

The int5 quantization collapse is under active investigation. We are exploring:

- **Mixed-precision quantization:** int8 for oscillator projections (5.6M params),
  int4 for FFN (10.5M params), fp16 for embeddings. Preliminary size estimates
  suggest this fits within 16 MB with careful compression tuning.
- **Early QAT:** Starting QAT from step 0 rather than the final phase, so the
  model learns to maintain quantization-friendly weight distributions throughout
  training.
- **Omega discretization:** Constraining oscillator frequencies to a fixed grid
  during training, reducing the precision required to represent them post-quantization.

If any of these pan out, we will update this submission.

---

## Appendix: Test-Time Training (TTT)

Not included in submission score. Per-document LoRA (rank=8, lr=0.01, chunk=256)
applied during eval. Score-then-train, causal, LoRA discarded between documents.

Applied to the fp32 model at 1860 steps (7L, val_bpb 1.4119 before TTT).
Result: 1.4119 -> 1.3789 bpb (-0.033). Gain is smaller than Transformer TTT
(~-0.10) because TRN layers have fewer LoRA-targetable projections.

---

## Acknowledgments

Thanks to OpenAI for the challenge and compute credits. We spent most of our
time debugging why int5 kept collapsing on overnight runs -- two 5-hour training
runs wasted before we identified the oscillator projection sensitivity. We are
still working on it. The 16 MB constraint forced us into quantization territory
we had never touched before, and we plan to keep poking at the TRN + low-bit
quantization problem after the deadline.
