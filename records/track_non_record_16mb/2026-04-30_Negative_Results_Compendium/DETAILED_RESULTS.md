# Negative Results Compendium: What Didn't Work on the Path to 1.063 BPB

**Author:** nprime06 | **Date:** 2026-04-30 | **Track:** Non-record (negative results)

This submission documents 14 experimental directions that failed, regressed, or produced only marginal gains during our work on the PR-1493 (1.0810 BPB) through PR-1787 (1.06335 BPB) frontier stacks. The experiments span training dynamics, quantization, architecture, data, optimizer, and systems — roughly 50+ compute runs on 8xH100 (Modal and RunPod).

We believe negative results are underrepresented in this competition. Every direction below consumed real GPU hours and produced real measurements. Some are definitive dead ends; others are merely marginal at this specific scale and might become relevant under different constraints.

**Our positive contribution (PR #1787, merged):** 1.06335 BPB (3-seed mean), stacking Polar Express NS, MIN_LR=0.10, sparse attention gate, fused softcapped CE, and PR #1767 TTT improvements on top of PR #1736.

---

## Table of Contents

1. [FP8 MLP Training](#1-fp8-mlp-training)
2. [Multi-Token Prediction (MTP)](#2-multi-token-prediction-mtp)
3. [Batch-Size Ramp](#3-batch-size-ramp)
4. [Weight Decay Scheduling](#4-weight-decay-scheduling)
5. [L1 Regularization for Sparsity](#5-l1-regularization-for-sparsity)
6. [Loop Depth Curriculum (1→2→3)](#6-loop-depth-curriculum-123)
7. [SparseGPT](#7-sparsegpt)
8. [Quantization Grid Design](#8-quantization-grid-design)
9. [Tokenizer Sweep](#9-tokenizer-sweep)
10. [Dataset Substitution (FineWeb-Edu)](#10-dataset-substitution-fineweb-edu)
11. [Systems Knobs (max-autotune)](#11-systems-knobs-max-autotune)
12. [DeepSeek NS10 Optimizer](#12-deepseek-ns10-optimizer)
13. [Weight Entropy and Distribution Shaping](#13-weight-entropy-and-distribution-shaping)
14. [Bigram Training (from NanoGPT)](#14-bigram-training-from-nanogpt)

---

## 1. FP8 MLP Training

**Verdict: Definitive negative. Net slowdown at all tested configurations.**

We investigated whether FP8 (`torchao.Float8Linear`) on MLP layers could buy training steps by reducing GEMM time. The motivation was straightforward: FP8 tensor cores on H100 offer 2x peak FLOPS over BF16. More steps in the 600s budget should improve pre-quant BPB.

### What we tried

| # | Config | Steps | Step time | Result |
|---|--------|------:|----------:|--------|
| 1 | BF16 baseline, MLP×2, 8xH100 | 1293 | 128.9 ms | Reference |
| 2 | FP8 tensorwise, MLP×2, 8xH100 | 1123 | ~+13% slower | **Net negative** |
| 3 | FP8 tensorwise (NCCL diag), 8xH100 | 1128 | ~+13% | Confirmed not NCCL-related |
| 4 | FP8 tensorwise, 1xH100 | 213 | ~+4% slower | Slightly worse even single-GPU |

The earlier small-scale probes (MLP×2 smoke tests) showed FP8 rowwise at **+18% slower** on simpler configs. All PR-1493-scale benchmarks used tensorwise scaling.

### Why it fails

Three independent reasons, each sufficient alone:

**1. Arithmetic intensity below the FP8 ridge.** Our MLP GEMMs (K=512, N=1024/2048) have arithmetic intensity of 340–407 FLOPs/byte. The H100 FP8 compute-bound ridge is ~591 FLOPs/byte. We're memory-bound in FP8, so the 2x FLOPS don't translate:

```
BF16 ridge:  ~295 FLOPs/byte  →  our GEMMs are compute-bound ✓
FP8 ridge:   ~591 FLOPs/byte  →  our GEMMs are memory-bound ✗
```

**2. amax + cast overhead exceeds savings ceiling.** Per MLP layer, FP8 requires ~6 memory-bound CUDA-core kernels (amax reductions + dtype casts) for dynamic scaling. Across 17 virtual MLPs × fwd+bwd:

```
amax+cast overhead:  ~17 ms/step  (measured via torch.profiler)
FP8 GEMM savings:    ~3-6 ms/step (ceiling, never achieved)
Net:                 +12 ms/step  (worse)
```

**3. MLP GEMM budget is small.** Even a perfect 2x FP8 GEMM speedup only saves half of ~5.8 ms/step (MLP×2). The overhead in reason 2 exceeds the theoretical maximum savings.

### Directions considered and rejected

- **Tensorwise scaling:** Reduces amax to single scalar, ~10-12 ms overhead. Still exceeds ceiling.
- **Delayed scaling (torchao ≤ 0.9):** Removed in torchao 0.10+. Known instability during LR transitions.
- **Input-side FP8 fusion (Variant B):** Estimated ~1.9% wall-clock gain at MLP×4 (~2.5 ms/step). Marginal ROI vs. other axes.
- **Full-stack FP8 (Stage 3):** The only path where FP8 wins decisively. Requires CUTLASS EVT FP8 kernels that don't exist in-repo. 1-2 weeks of kernel work, not viable in competition timeline.

### When it becomes relevant

FP8 flips positive if model_dim grows past ~1024 or MLP hidden past ~3072 (crossing the FP8 ridge), or when fused Triton/CUTLASS kernels eliminate the amax+cast overhead. Both are currently blocked by the 16 MB artifact cap.

---

## 2. Multi-Token Prediction (MTP)

**Verdict: Overhead too high at this scale. Three separate implementations all negative.**

We tested whether multi-token prediction could improve the training signal per step. Three implementations, all on 8xH100 with 600s budgets:

### Implementation A: Untied low-rank MTP head (PR-1493 base)

A factorized auxiliary head `hidden(512) → rank(64) → vocab(8192)` = 557K params (13.3% of a full head). Separate AdamW optimizer group. MTP loss weight 0.05.

```python
# The auxiliary head
class MTPHead(nn.Module):
    def __init__(self, dim, rank, vocab_size):
        self.down = CastedLinear(dim, rank)
        self.up = CastedLinear(rank, vocab_size)
    def forward(self, x):
        return self.up(self.down(x))

# Loss: next-token + weighted future-token
loss = ce_loss + mtp_weight * ce(mtp_head(hidden), target_shifted)
```

**Result:** 673 steps in 120s smoke test, pre-quant 1.323 BPB. Severe throughput loss from the auxiliary forward/backward. The extra params also compete with the 16 MB artifact budget.

### Implementation B: NanoGPT-style shared-logit MTP (PR-1787 base)

No auxiliary heads — reuses the same LM-head logits for future targets:

```python
# Shared-logit MTP: same logits, different targets
loss = w0 * CE(logits_i, target_i)
     + w1 * CE(logits_i, target_{i+1})
     + w2 * CE(logits_i, target_{i+2})

# Default weights: [1.0, 0.5, 0.25], decayed via NanoGPT schedule
# w2 decays over first 1/3 of training
# w1 decays over second 1/3
# Only w0 remains in final 1/3
```

We implemented three loss backends:
- `fused_mtp`: Custom Triton kernel computing one denominator + weighted future-target subtraction
- `target_delta`: Reuses existing fused CE + selected-logit corrections
- `multi_ce`: Straightforward one CE call per horizon

**Result:** `target_delta` lost ~4% step-rate from gather/tanh autograd. `fused_mtp` recovered most of it but the overall MTP signal was noise-level on pre-quant BPB. The auxiliary loss adds gradient noise without enough signal to compensate for the lost training tokens (fewer effective tokens per step due to overhead).

### Implementation C: NanoGPT-style MTP on PR-1953 stack

Same shared-logit design ported to the latest clean stack. Same conclusion — the competition's 600s budget is too tight for the MTP signal to pay for its overhead.

### Why MTP fails here

The core issue is budget: MTP adds ~4-8% overhead per step (even fused), which at 600s translates to ~200-400 fewer training steps. At 36M params with ~5000 steps, each step matters. The future-token prediction signal is too weak to compensate for the lost steps. This matches PR #1272's comprehensive negative results on "strong model" augmentations.

---

## 3. Batch-Size Ramp

**Verdict: Marginal quality win (-0.0012 BPB), eaten by compilation overhead.**

From GPT-2 / NanoGPT-speedrun literature: smaller batch early provides high gradient variance that escapes shallow minima (McCandlish 2018 critical-batch-size framing), then full batch late ensures clean convergence.

### Design

Linear ramp from 131,072 → 786,432 tokens/step over the first 20% of wallclock:

```python
if frac < BATCH_RAMP_END_FRAC:
    ramp_frac = frac / BATCH_RAMP_END_FRAC
    target_tokens = int(start_tokens + ramp_frac * (full_tokens - start_tokens))
    # Round down to nearest world_size * grad_accum * seq_len
    effective_tokens = (target_tokens // granularity) * granularity
```

Discrete 4-stage variant (avoids continuous recompilation):

| Stage | Frac range | Seqs/GPU | Tokens/step |
|-------|-----------|----------|-------------|
| 1 | 0.00–0.05 | 8 | 131,072 |
| 2 | 0.05–0.10 | 16 | 262,144 |
| 3 | 0.10–0.15 | 32 | 524,288 |
| 4 | 0.15–1.00 | 48 | 786,432 |

### Results (PR-1493 base, seed 42, 8xH100)

| Config | Pre-quant BPB | Post-quant BPB | Notes |
|--------|---------------|----------------|-------|
| Stock (no ramp) | 1.0872 | 1.0984 | Reference |
| Continuous ramp | 1.0862 | 1.0974 | -0.0010 pre, compilation transitions |
| **Discrete 4-stage** | **1.0860** | **1.0972** | **-0.0012 pre, -0.0010 post** |
| Discrete + `dynamic=True` | 1.0869 | — | `dynamic=True` compile cost +0.0007 |

### Why we dropped it

1. **Token dilution:** Small-batch stages process fewer tokens per wallclock second. The ramp saves ~200 steps of "wasted" large-batch early training but loses ~150 steps of raw token throughput.
2. **Compilation overhead:** Each batch-size transition triggers `torch.compile` graph recompilation. Even with `cache_size_limit=64` and discrete stages, this costs ~15-30s of the 600s budget.
3. **Interaction with PR-1736/1787 stack:** When ablated on the PR-1736 base, batch ramp showed **+1.94 mBPB** (a loss), likely because the deeper stack's training dynamics are better-tuned to fixed batch from step 1.

The isolated signal (-0.0012 on PR-1493) is real but fragile across base stacks.

---

## 4. Weight Decay Scheduling

**Verdict: Overall a wash. L2 WD is scale-invariant under SDClip — changing WD scale cannot change artifact size.**

This is one of the most important structural insights from our work.

### The scale-invariance theorem

SDClip quantization computes per-row scales:

```python
scale = k * std(row) / clip_range
q = round(w / scale).clamp(-clip_range, clip_range)
```

Scaling all weights by a constant γ:
```
w → γw
std(row) → γ·std(row)
scale → γ·scale
q = round(γw / γ·scale) = round(w / scale)  # unchanged!
```

**L2 weight decay shrinks weights proportionally. SDClip absorbs this into the per-row scale. The quantized integers — and therefore the compressed artifact — are identical.**

### What we tried

| Config | Pre-quant BPB | Post-quant BPB | Artifact MB | Notes |
|--------|---------------|----------------|-------------|-------|
| Stock WD | 1.0872 | 1.0984 | 15.97 | Reference |
| MUON_WD=0.15 (fixed) | 1.0890 | 1.0989 | **15.97** | Weights 29.4% smaller RMS. **Artifact identical.** |
| WD taper 0.095→0.048 | **1.0868** | **1.0982** | **15.97** | Quality improves. Artifact unchanged. |
| Binary WD flip (off early, on late) | 1.0930 | 1.1060 | — | **+0.0058 pre-quant.** Early WD is load-bearing. |
| Scalar WD zero | ≈ baseline | ≈ baseline | ≈ baseline | ±0.001 BPB, neutral |

### The WD flip was the most informative

We tested removing WD entirely during the first 28% of training (matching the LR warmdown boundary), then restoring stock WD:

```python
wd_coef = stock_wd * (0.0 if frac < 0.28 else 1.0)
```

**Result: +0.0058 pre-quant, +0.0076 post-quant.** The quant gap also widened (+0.013 vs +0.011). Early WD is load-bearing for this architecture — it constrains the optimizer trajectory into a region that quantizes well. Don't repeat the binary-off probe.

### Takeaway

WD should be tuned for **model quality**, not compression. The artifact is structurally invariant to proportional scale changes. Late WD tapering (PR-1729 style) is a small quality win (-0.0002 BPB) at zero artifact cost.

---

## 5. L1 Regularization for Sparsity

**Verdict: Cumulatively unstable at 36M params. All L1 variants either crashed or cost more BPB than they saved.**

The idea: L1 changes the weight distribution **shape** (not just scale), which could break the SDClip scale-invariance barrier that makes L2 ineffective for compression.

### What we tried

```python
# Proximal L1 during warmdown phase
for p in mlp_params:
    threshold = l1_lambda * lr_scale  # or fixed
    p.data = torch.sign(p.data) * torch.relu(torch.abs(p.data) - threshold)
```

| Config | Pre-quant BPB | Artifact | Notes |
|--------|---------------|----------|-------|
| L1 fixed λ=0.0001 | **1.3651** | — | **Crashed at step 4500.** Cumulative L1 overwhelmed model as LR decayed. |
| L1 LR-scaled λ=0.0001 | **1.3780** | — | **Same crash.** LR scaling didn't prevent cumulative damage. |
| L1=0.00001 + floor 2% + WD ramp→0.3 | 1.0943 | ~15.9 MB | Stable but **+0.007 BPB.** WD ramp too aggressive. L1 too weak to create sparsity. |

### Why L1 fails here

At 36M params over ~4500 steps of active L1, each proximal step is a global perturbation. Even tiny λ accumulates — the model loses important weight structure faster than the optimizer can recover. The crash pattern is characteristic: training looks fine for ~3000 steps, then loss spikes and never recovers as LR decays and can no longer compensate for the accumulated L1 damage.

The post-hoc pruning experiments tell the same story from the other side: even optimal (Hessian-aware) pruning at 30% costs +0.021 BPB. There's minimal redundancy at this model size.

---

## 6. Loop Depth Curriculum (1→2→3)

**Verdict: Undertrained. Loses 1.15 mBPB on the PR-1736 base.**

PR-1756 (1.06505 BPB) uses a training curriculum that activates recurrence depth gradually:

```python
TRAIN_LOOP_PHASE_DEPTHS = [1, 2, 3]
TRAIN_LOOP_PHASE_FRACTIONS = [0.35, 0.30, 0.35]
# Phase 1: depth=1 for first 35% of training
# Phase 2: depth=2 for next 30%
# Phase 3: depth=3 for final 35%
```

When ablated on the PR-1736 base in our 7-knob sweep, this showed **+1.15 mBPB** vs stock binary 1→3 at frac=0.35.

The likely mechanism: each phase trains a different effective architecture. Phase 1 (depth-1) gets 35% of the token budget but produces representations that depth-2 and depth-3 must then adapt to. The final depth-3 phase gets only 35% of tokens — not enough to fully train the recurrent path. Stock PR-1736's binary 1→3 gives the depth-3 path 65% of tokens, which is apparently better.

This might work at longer training budgets where the undertrained-phase problem is less severe, but at 600s / ~5000 steps it's a net loss.

---

## 7. SparseGPT

**Verdict: Too expensive for the compression it delivers. k-inflation is strictly better.**

SparseGPT integrates sparsification into the GPTQ Cholesky sweep: for each column, it decides whether to zero the weight (sparse) or quantize it, based on a cost threshold:

```python
# During GPTQ column sweep:
zero_cost = w_col**2 / H_diag[col]      # cost of zeroing
quant_cost = (w_col - q_col)**2 / H_diag[col]  # cost of quantizing
if zero_cost < threshold * quant_cost:
    q_col = 0  # sparse
```

### Results (PR-1493 base)

| Threshold | Sparsity | Post-quant BPB | BPB cost | Artifact MB saved |
|-----------|----------|----------------|----------|-------------------|
| 1.0 | 0% | 1.0984 | 0 | 0 |
| 2.0 | ~20% | 1.0988 | +0.0004 | 0.11 |
| 5.0 | ~23% | 1.1004 | +0.002 | 0.24 |
| 10.0 | ~27% | 1.1064 | +0.008 | 0.38 |

Compare with simple k-inflation:

| k | Post-quant BPB | BPB cost | Artifact MB saved |
|---|----------------|----------|-------------------|
| 12.85 (stock) | 1.0984 | 0 | 0 |
| **19.3** | **1.1124** | **+0.014** | **2.25** |

**k=19.3 saves 6x more bytes than SparseGPT at comparable BPB cost.** The reason: Brotli already compresses near-zero integers almost as well as exact zeros. SparseGPT finds safely-zeroable weights, but the compression benefit of `0` vs `±1` or `±2` is negligible under Brotli-11. Meanwhile, k-inflation pushes the entire distribution peakier, which Brotli loves.

SparseGPT combined with k-inflation (`k=19.3 + t=2`) saved only 0.07 MB more than k=19.3 alone — confirming that wider bins already push the near-zero weights to exact zero.

---

## 8. Quantization Grid Design

**Verdict: Uniform int6 with SDClip is already near-optimal. Non-uniform grids and alternative bit-widths are dominated.**

We tested extensively whether the int6 uniform grid could be improved:

### Non-uniform grids (NF5)

NF5 maps weights to 32 non-uniform levels optimized for Gaussian distributions:

| Config | Post-quant BPB | Artifact MB |
|--------|----------------|-------------|
| int6 k=12.85 (stock) | 1.0984 | 15.97 |
| **NF5** | **1.1112** | **22.43** |

NF5 is worse on **both axes**. The levels have maximum entropy by construction (equal-probability bins for Gaussian input), which is exactly what Brotli hates. The stock int6 grid at k=12.85 is tuned for Brotli compression, not MSE — and that's the right tradeoff.

### int5

| Config | Post-quant BPB | Artifact MB |
|--------|----------------|-------------|
| int5 k=6 | 1.0983 | 16.18 |
| int5 k=12.85 | 1.1398 | 12.02 |

int5 at k=6 achieves the same quality as int6 at k=12.85 (bin width is nearly identical: 0.39σ vs 0.41σ), but the artifact is 175 KB over the cap. int5 at stock k collapses quality. The model is effectively a ~3.56 bits/value model after Brotli regardless of container format.

### Per-tensor bit allocation

| Config | BPB cost | Artifact saved |
|--------|----------|----------------|
| Tiered (NF3/int5/int6 by sensitivity) | +0.003 | 0 MB net |
| Tiered k (int5 k=6 sensitive, k=12.85 tolerant) | +0.016 | 1.13 MB |

**GPTQ's Hessian error compensation absorbs most allocation differences.** Rearranging the same total bits across tensors barely moves BPB because GPTQ propagates quantization error optimally within each tensor.

### Entropy-constrained and central-bucket GPTQ

We modified GPTQ rounding to prefer central buckets:

```python
# Standard GPTQ: minimize squared error
score = (w - q)**2 / H_diag

# Central-bucket GPTQ: add abs(q) penalty
score = (w - q)**2 / H_diag + lambda * abs(q)
```

Best result at λ=0.005: -0.029 mBPB and 1.3 KB saved. Real but tiny — useful only as boundary polish near the 16 MB cap.

### Permutation ordering

We exploited the MLP hidden-unit permutation symmetry (reordering `mlp.fc` rows and `mlp.proj` columns preserves the float model) to improve Brotli's byte-level autocorrelation:

```python
# Sort MLP hidden units by L2 norm of the upstream (fc) row
perm = torch.argsort(mlp_fc.weight.norm(dim=1))
mlp_fc.weight.data = mlp_fc.weight[perm]
mlp_proj.weight.data = mlp_proj.weight[:, perm]
```

Best result: 12 KB saved, 0 BPB cost. Free but small — 0.075% of the artifact.

---

## 9. Tokenizer Sweep

**Verdict: No more CaseOps-like opportunities exist.**

CaseOps (PR-1729) was the last major tokenizer innovation: a bijective case-folding transform that merges upper/lowercase tokens, saving ~8% of vocab slots. We audited whether similar byte-level transforms could further conserve tokens.

Candidates evaluated:
- **Digit normalization** (map all digits to a canonical digit): breaks semantic meaning for numbers
- **Whitespace normalization** (merge whitespace variants): already handled by SentencePiece
- **Punctuation folding** (merge quote variants, dash variants): too few tokens saved vs. information loss
- **Unicode normalization** (NFC/NFKD): SentencePiece already normalizes

The SP1024 → SP4096 → SP8192 progression captured the major vocab-size gains. CaseOps was the last transform where the byte-savings-per-information-loss ratio was favorable. The remaining tokenizer headroom is in sub-word segmentation tuning, which is a much weaker lever.

PR #1271 (Scylla tokenizer) is a cautionary tale: 93% of its claimed BPB gap was a byte-accounting error. Tokenizer work is high-risk for subtle measurement bugs.

---

## 10. Dataset Substitution (FineWeb-Edu)

**Verdict: FineWeb-Edu is a bad direction. Moving toward the original train distribution consistently improves BPB.**

We tested whether a "cleaner" training corpus (FineWeb-Edu) could help under the 600s token budget:

| Config | Post-quant BPB | Δ vs original |
|--------|----------------|---------------|
| **Original challenge data** | **1.0829** | **reference** |
| 100% FineWeb-Edu | 1.1507 | +0.068 |
| 50% original + 50% Edu | 1.0957 | +0.013 |
| 70% original + 30% Edu | 1.0887 | +0.006 |
| 70/30 document-level mix | 1.0885 | +0.006 |

The results are monotonic: more original data = better BPB. The challenge val set is drawn from the same distribution as the original train data, so any distribution shift hurts.

Document-level mixing was slightly better than token-chunk mixing, but the delta was tiny (0.0002 BPB).

**Caveat:** Our alternate datasets used ~1B unique train tokens with ~3.6x data reuse, vs. the original's ~8B tokens. This amplifies distribution mismatch effects. But the direction is clear — don't substitute the training data.

---

## 11. Systems Knobs (max-autotune)

**Verdict: Default torch.compile choices are already near-optimal for these GEMM shapes.**

We tested `torch.compile(mode="max-autotune-no-cudagraphs")` on both the naive baseline and PR-1493:

### Naive baseline (root train_gpt.py)

| Config | Steps (90s) | Step time | Compile overhead |
|--------|-------------|-----------|-----------------|
| Default compile | 2078 | 43.24 ms | ~290s launch |
| max-autotune | 4145 (180s) | **43.37 ms (+0.3%)** | **~811s launch (+521s)** |

**+0.3% slower steady-state, +521s of autotuning on cold container.** The autotune benchmarks ~200 candidate kernels per shape, finding marginally-faster split-K variants that don't move end-to-end time.

### PR-1493 SOTA stack

| Config | Steps | Step time | Compile overhead |
|--------|-------|-----------|-----------------|
| Default compile | ~4562 | 128.9 ms | Standard |
| max-autotune | 1314 (168s) | **127.87 ms (-0.8%)** | **~2373s launch (+34 min)** |

**-0.8% faster steady-state (~1 ms/step), but +34 minutes of compile overhead on cold container.** The 600s training budget makes this completely unviable without persistent `TORCHINDUCTOR_CACHE_DIR` infrastructure (which the challenge doesn't provide).

The finding cross-validates PR #1679's conclusion that Inductor-tuning levers are dead at this scale.

---

## 12. DeepSeek NS10 Optimizer

**Verdict: Slightly slower, marginal BPB win that doesn't hold up.**

DeepSeek V4's hybrid Newton-Schulz uses 10 steps (8 aggressive + 2 stabilizer) vs. the standard 5-step or Polar Express 5-step:

```python
# Standard 5-step NS (Polar Express coefficients):
# Each step: X = a*X + b*X@X@X + c*X@X@X@X@X
# 5 fixed (a,b,c) tuples optimized via minimax

# DeepSeek 10-step NS:
# 8 aggressive steps (larger step sizes) + 2 stabilizer steps
# Higher-quality polar factor per optimizer step
```

### Result (1-seed, PR-1493 base)

Pre-quant BPB: **1.06664** vs Polar Express 5-step **1.06764** = **-1.0 mBPB**.

However, the 10-step NS is slightly slower per optimizer step (more matrix multiplications), and subsequent runs on the combined stack suggested the win was within noise of the Polar Express result. Since Polar Express gives comparable quality at 5 steps with zero throughput cost, NS10 is dominated.

This matches the general pattern: Newton-Schulz orthogonalization quality saturates quickly. The 5-step Polar Express coefficients (minimax-optimized per-iteration tuples) already capture most of the useful polar factor quality.

---

## 13. Weight Entropy and Distribution Shaping

**Verdict: The most theoretically interesting negative result. Gaussian weights are simultaneously maximum-entropy (worst for compression) and maximum-expressiveness (best for the model). Simple regularizers cannot resolve this tension.**

This was our deepest investigation, spanning 6+ training runs and extensive post-training analysis. A separate detailed document (`WEIGHT_ENTROPY_ANALYSIS.md`) is included.

### The core insight

The competition's compression pipeline is:

```
trained weights → GPTQ int6 → Brotli → artifact bytes
```

The compressed artifact size is governed by the Shannon entropy of the quantized integer stream. Stock PR-1787 MLP weights match Gaussian bucket predictions almost exactly (note: the PR-1493 baseline had a measured Shannon entropy of 3.327 bits/symbol at k=12.85; the PR-1787 values below differ because the trained model and k settings changed):

| Metric | Gaussian prediction | Empirical stock MLP |
|--------|--------------------:|--------------------:|
| Entropy | 3.425 bits/symbol | 3.427 bits/symbol |
| Zero mass | 15.3% | 15.5% |
| \|q\|≤2 | 66.7% | 67.0% |
| \|q\|≥8 | 0.4% | 0.4% |

Brotli already achieves 101.3% of Shannon on this stream (slightly exceeds the order-0 bound via context modeling). **There is no entropy-coder headroom.**

The key mathematical fact: **Gaussian is the maximum-entropy distribution at fixed variance.** L2 regularization induces near-Gaussian weights. This means L2 training makes weights maximally hard to compress at their given scale — but it also means the weights have maximum information capacity per parameter, which is exactly what makes them good for modeling.

### What we tried

**1. Kurtosis penalty** — reduce excess kurtosis to make the distribution "less tailed than Gaussian":

```python
z = w_row - mean(w_row)
kurtosis = E[z^4] / E[z^2]^2
penalty = mean(relu(kurtosis - target)^2)
# target < 3.0 (Gaussian kurtosis)
```

**Result:** Kurtosis moved, but bucket entropy didn't. The compressor sees discrete q buckets after row scaling and GPTQ, not continuous kurtosis. Moving fourth moments can leave the central bucket distribution unchanged.

**2. Shoulder penalty** — directly penalize values in the MLP shoulder region |q|=4..6:

```python
q_soft = (w_row - mean(w_row)) / std(w_row) * (clip_range / k)
shoulder_mask = (abs(q_soft) > 3.5) & (abs(q_soft) < 6.5)
penalty = mean(shoulder_mask.float() * (abs(q_soft) - 3)**2)
```

| Config | Post-quant BPB | Artifact bytes | Δ bytes |
|--------|----------------|----------------|---------|
| Late WD-off only | 1.07614 | 15,908,462 | -7 |
| Shoulder λ=1e-3 + WD-off | 1.07682 | 15,906,234 | **-2,235** |
| Shoulder λ=1e-2 early + WD-off | **1.07602** | 15,911,521 | +3,052 |

The BPB win in the aggressive shoulder run came from better pre-quant quality (late WD removal), **not** from entropy shaping. Exact GPTQ bucket audit showed main-int6 entropy moved by only ±0.0002 bits/symbol.

**3. Q-center regularizer** — strong center pressure on row-normalized q-coordinates:

```python
x = w / std(row) * (clip_range / k)
loss += lambda * mean((abs(x) - center_target)**2)
```

| Run | Post BPB | Artifact bytes | MLP H (bits) | MLP q0 | MLP \|q\|≤3 |
|-----|----------|----------------|-------------|--------|------------|
| Stock | 1.07677 | 15,908,469 | 3.427 | 15.5% | 82.6% |
| Q-center | 1.07842 | **15,649,631** | **3.354** | 14.5% | 85.9% |

**This proved training-time entropy shaping CAN move the final q stream.** It saved 259 KB. But it cost +1.64 mBPB because it thinned the zero spike and pushed mass into high tails. The entropy win was real but came from a shape the model couldn't tolerate.

**4. Q-transport regularizer** — constrained transport: move only the |q|=5/6 shoulder toward |q|=3/4, protect central mass:

```python
# Only penalize shoulder region, protect center and cap tails
shoulder = sigmoid_band(abs(q_soft), lo=4.5, hi=6.5)
penalty = shoulder * (abs(q_soft) - 3)**2
tail_penalty = softplus(abs(q_soft) - 7)**2
center_floor = relu(target_q0 - q0_mass)**2
loss += lambda * (mean(penalty) + tail_wt * mean(tail_penalty))
     + floor_wt * center_floor
```

| Run | Post BPB | Artifact bytes | Δ BPB | Δ bytes |
|-----|----------|----------------|-------|---------|
| Conservative q-transport | 1.07696 | 15,803,340 | +0.185 mBPB | **-105,129** |
| Strong q-transport | 1.07888 | 15,584,406 | +2.11 mBPB | -324,063 |

Conservative q-transport is the best training-time entropy result: 105 KB saved for only +0.185 mBPB. But the strong variant revealed the failure mode — the loss mask had loopholes. Values moved below the mask boundary and settled at q=4 instead of q=3. The tail budget constrained |q|≥7 as a lump but didn't prevent |q|≥8 growth. The model games the objective.

### Why the strategy may be fundamentally limited

1. **Local bucket penalties are easy to game.** The model has many degrees of freedom. Penalizing one q-region pushes mass to neighboring unpenalized regions that still compress badly.

2. **The proxy is not the saved GPTQ stream.** Training-time diagnostics use row-normalized hard-rounded proxies. GPTQ with Hessian-aware error compensation partially rewrites the final q stream.

3. **Entropy alone is not quality-aware.** The most compressible distribution may be a bad model. The model needs some apparent entropy in the weights to represent useful functions.

4. **The stock distribution is already near a local optimum.** 82.6% of values in |q|≤3, only 0.44% in |q|≥8. There isn't much tail mass to remove. Large wins require moving the 16.1% shoulder mass without damaging reconstruction — and that appears harder than simple penalties assumed.

5. **The best wins come from allocation, not pure entropy.** Loop-aware k scheduling (giving more precision to recurrent layers) produced the strongest reliable quant-only improvements. This suggests the right direction is "use bytes where sensitivity is high, save bytes where sensitivity is low" — not "make the whole q stream lower entropy."

### The reliable quant levers we found

| Lever | Description | Δ BPB | Δ bytes |
|-------|-------------|-------|---------|
| Loop-aware k | LOOP_MLP=11, LOOP_ATTN=12.85, stock elsewhere | -0.45 mBPB | +91 KB |
| LQER top-1 rank-4 | Low-rank residual on tok_emb only | -0.21 mBPB | +4.3 KB |
| Central-bucket GPTQ λ=0.02 | Penalty for large |q| during rounding | ~0 mBPB | -5.2 KB |
| MLP tensor-scale | One scale per tensor instead of per-row | +0.14 mBPB | -32.6 KB |

These are modest but reliable. The best measured under-16MB quant-only config is loop-aware k + LQER top-1: **1.07632 BPB at 15.9995 MB** (vs stock 1.07677 at 15.908 MB).

---

## 14. Bigram Training (from NanoGPT)

**Verdict: Removed from SOTA stacks. At SP8192 vocab, the model natively captures bigram information.**

BigramHash was a core component of early records (PR #65 through PR #1019), providing a learnable hash-table embedding for character bigrams. The hash size evolved non-monotonically: 4096 → 10240 → 2048 → 3072.

At the SP8192 frontier, BigramHash was dropped from the SOTA lineage. The larger vocabulary already captures most useful bigram information through subword co-occurrence patterns. PR #1420 found that n-gram tilt (biasing the loss toward n-gram-predictable tokens) was more effective than maintaining a separate BigramHash embedding.

We did not independently validate the removal, but the consistent pattern across PR #1394 → PR #1736 → PR #1787 is that no frontier submission uses BigramHash. The artifact budget pressure at SP8192 (embedding table alone is ~4 MB pre-compression) makes auxiliary embedding tables hard to justify.

---

## Summary: What the Competition Teaches About Small-Model Training

The 16 MB / 600s constraint creates a unique optimization landscape where many standard deep learning techniques fail:

1. **The artifact cap dominates.** Most "training improvements" that work in unconstrained settings (FP8, MTP, larger batch, data augmentation) fail because they either cost throughput in a fixed time budget or add parameters that don't fit in 16 MB.

2. **Compression is the binding constraint, not model quality.** The pre-quant → post-quant gap (~0.01 BPB) is small relative to the absolute BPB (~1.07). The real bottleneck is fitting the quantized model into 16 MB while preserving as much quality as possible.

3. **Gaussian weights are a fundamental tension.** L2 regularization makes weights maximally expressive (maximum entropy at fixed variance) but also maximally hard to compress. This is not a bug — it's a feature. The model is using all available information capacity.

4. **GPTQ absorbs most naive quantization improvements.** Per-tensor bit allocation, sensitivity-based reallocation, and alternative grids all get absorbed by GPTQ's Hessian error compensation. The strongest quant lever is the simplest: k-inflation (widen the clipping range to push the integer distribution peakier).

5. **The only reliable axes are (a) base model quality and (b) TTT.** Every 1 mBPB improvement in pre-quant quality translates ~1:1 through quantization and TTT. TTT provides a roughly constant ~13 mBPB gain. Everything else is boundary polish.

## Acknowledgments

Infrastructure: Modal (8xH100 on-demand), RunPod (8xH100 persistent). Base stacks: PR #1493 (bigbag), PR #1736 (dexhunter), PR #1787 (ours). The competition organizers at OpenAI for creating a challenge that rewards rigorous experimentation.
