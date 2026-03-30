# Technical Report: 11L XSA + Score-First LoRA TTT

**val_bpb: 1.1573** | **15.02 MB** artifact | 1xH100 PCIe, ~80 min | 26.5M parameters

This submission represents 18 experiments over 5 days, improving from val_bpb=3.10 to 1.1573 through systematic architecture exploration, quantization optimization, and test-time adaptation. Total compute cost: ~$50.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Training Configuration](#2-training-configuration)
3. [Techniques](#3-techniques)
   - [3.1 SmearGate](#31-smeargate)
   - [3.2 SwiGLU 3x MLP](#32-swiglu-3x-mlp)
   - [3.3 U-Net Skip Connections](#33-u-net-skip-connections)
   - [3.4 XSA (Cross-Sequence Attention)](#34-xsa-cross-sequence-attention)
   - [3.5 Orthogonal Initialization](#35-orthogonal-initialization)
   - [3.6 Muon Optimizer + Weight Decay](#36-muon-optimizer--weight-decay)
   - [3.7 Stochastic Weight Averaging](#37-stochastic-weight-averaging)
   - [3.8 Mixed Quantization](#38-mixed-quantization-int5int6int8--zstd)
   - [3.9 Score-First LoRA TTT](#39-score-first-lora-ttt)
4. [Training Progression](#4-training-progression)
5. [What Each Technique Contributed](#5-what-each-technique-contributed)
6. [What Didn't Work](#6-what-didnt-work)
7. [Bits Budget Analysis](#7-bits-budget-analysis)
8. [Layer & Head Ablations](#8-layer--head-ablations)
9. [Experiment Journey](#9-experiment-journey)
10. [Comparison to Competition](#10-comparison-to-competition)
11. [Conclusion](#11-conclusion)

---

## 1. Architecture Overview

```
Input tokens (1024 vocab)
    ↓
Token Embedding (1024 × 512, tied) → int8 quantized
    ↓
SmearGate (blend with previous token embedding)
    ↓
RMSNorm
    ↓
┌──────────────────────────────────────────────┐
│  11 Transformer Blocks (U-Net)               │
│                                              │
│  Encoder (L0-L4):   save skip outputs        │
│  Bottleneck (L5):   U-Net crossover          │
│  Decoder (L6-L10):  add weighted skips back  │
│                                              │
│  Each block:                                 │
│    resid_mix → RMSNorm → Attn → scale →      │
│    RMSNorm → SwiGLU MLP → scale              │
│                                              │
│  Attention: 8 heads, 4 KV heads (GQA)       │
│    L7-L10: XSA (subtract self-value proj)    │
│                                              │
│  MLP: SwiGLU 3x (gate·up→proj)              │
│    Hidden dim: 1024 (rounded to 64)          │
│                                              │
│  OrthoInit on all weights ≥64 dims           │
└──────────────────────────────────────────────┘
    ↓
RMSNorm → Tied embedding projection → logit softcap (30.0)
    ↓
Output probabilities
```

**Parameters**: 26,502,232 total | **Artifact**: 15.02 MB (15,793,319 bytes)

---

## 2. Training Configuration

| Parameter | Value |
|---|---|
| GPU | 1xH100 PCIe (RunPod) |
| Wallclock | ~4,850s (~80 min) |
| Batch size | 524,288 tokens/step (grad_accum=8) |
| Sequence length | 2,048 |
| Steps completed | 7,926 / 20,000 (wallclock cap) |
| Matrix LR | 0.04 |
| Scalar LR | 0.04 |
| Tied embed LR | 0.05 |
| Optimizer | Muon (matrices) + Adam (scalars) |
| Muon momentum | 0.95 (warmup from 0.85 over 500 steps) |
| Weight decay | 0.04 (decoupled, matrices only) |
| Warmdown | 3,000 iterations (cosine decay) |
| SWA | Every 200 steps during warmdown |
| Precision | bfloat16 (autocast) |
| Peak memory | 18,476 MiB |
| Avg step time | 612 ms |

---

## 3. Techniques

### 3.1 SmearGate

**What**: Blends each token's embedding with the previous token's embedding via a learned per-dimension gate.

```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        g = sigmoid(self.gate)[None, None, :]
        x_prev = cat([zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```

**Why**: Gives immediate bigram context at the embedding layer, freeing attention heads from doing local token-pair lookups. With a 1024-token vocabulary, knowing the previous token dramatically narrows the prediction space (e.g., after `▁th` the next token is almost certainly `e`).

**Impact**: ~0.012 BPB. Without SmearGate: +1.80 loss (catastrophic). The model completely internalizes the previous-token blend signal.

**Cost**: 512 learnable gate parameters. Zero step-time overhead.

**Verdict**: Non-negotiable. Requires OrthoInit to work — without orthogonal initialization, SmearGate hurts performance.

---

### 3.2 SwiGLU 3x MLP

**What**: Replaces ReLU² MLP with a gated activation: `swish(gate(x)) * up(x)`, with 3x expansion ratio.

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        hidden = int(2 * mlp_mult * dim / 3)  # rounded to 64
        self.gate = Linear(dim, hidden)
        self.up   = Linear(dim, hidden)
        self.proj = Linear(hidden, dim)  # zero-init

    def forward(self, x):
        return self.proj(silu(self.gate(x)) * self.up(x))
```

**Why**: SwiGLU's gating mechanism enables layers to selectively pass features. With ReLU², the output layer (L10) had impact 2.92; with SwiGLU, L10 reaches 3.15 and previously near-dead L8 jumps from 0.11 to 0.44 — "awakened" by gating.

**Impact**: +0.004 BPB over ReLU² at matched parameter count.

**Hidden dim**: 1024 (from `int(2 * 3 * 512 / 3)` rounded to nearest 64).

**Verdict**: Small but consistent. More importantly, it activates dead layers and improves layer specialization.

---

### 3.3 U-Net Skip Connections

**What**: Encoder-decoder architecture with learnable skip connections between mirrored layers.

```python
# Encoder (L0-L4): save skip outputs
for i in range(num_encoder_layers):
    x = blocks[i](x, x0)
    skips.append(x)

# Decoder (L6-L10): add weighted skip inputs
for i in range(num_decoder_layers):
    if skips:
        x = x + skip_weights[i] * skips.pop()
    x = blocks[num_encoder_layers + i](x, x0)
```

**Why**: Skip connections distribute gradient flow evenly, preventing dead layers in deeper networks. In a 9-layer model without U-Net, layers 4-6 had ablation impact of only 0.10-0.14 (near-dead). With U-Net on 11 layers, the weakest layer (L8) has 0.19 impact — zero dead layers.

**Impact**: Enables 11 layers without dead layer waste. Each block also receives `x0` (initial embedding) for an additional residual pathway.

**Verdict**: Essential for going beyond 9 layers. Learnable `skip_weights` (per-dimension, one per decoder layer) let the model control skip contribution.

---

### 3.4 XSA (Cross-Sequence Attention)

**What**: Removes the self-value projection from attention output in the last 4 layers (L7-L10).

```python
if self.use_xsa:
    # Subtract projection of y onto v (self-value component)
    dot_yv = (y * v).sum(dim=-1, keepdim=True)
    dot_vv = (v * v).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    y = y - (dot_yv / dot_vv) * v
```

**Why**: Forces attention to contribute only cross-position context, not redundant self-information. In standard attention, the output `y = Σ_j attn[j] * v[j]` includes `attn[self] * v[self]`, which is just the token's own value passed through — wasteful. XSA subtracts this component.

**Impact**: +0.005 BPB, consistent across training checkpoints (-0.003 to -0.006 at each).

**Applied to**: Last 4 of 11 layers, where self-attention bias is strongest.

**Cost**: ~7% step-time overhead for the projection computation.

**Verdict**: Reliable improvement. Short 300-step tests showed only noise (-0.0007), but full-length validation confirmed the effect. Sub-0.01 BPP improvements require full-length runs to measure.

---

### 3.5 Orthogonal Initialization

**What**: All Linear weight matrices ≥64 dims initialized orthogonally. Projection layers (attention out, MLP out) additionally scaled by `1/√(2 * num_layers)`.

```python
if module.weight.ndim == 2 and min(module.weight.shape) >= 64:
    nn.init.orthogonal_(module.weight, gain=1.0)
    if ".proj." in name:
        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```

**Why**: Orthogonal matrices preserve gradient norms through deep networks. The projection scaling (`1/√22 ≈ 0.213`) prevents gradient explosion through the U-Net's skip connections. Required for SmearGate to function — without it, the gate's learned blending destabilizes training.

**Impact**: ~0.002-0.003 BPB directly; enables SmearGate (+0.012 BPB).

**Verdict**: Foundation technique. Makes everything else work.

---

### 3.6 Muon Optimizer + Weight Decay

**What**: Muon optimizer for 2D weight matrices (zero-power operator via Newton-Schulz), Adam for 1D scalars. Decoupled weight decay of 0.04 on matrices only.

**Muon config**:
- LR: 0.04 | Momentum: 0.95 (warmup from 0.85 over 500 steps)
- Newton-Schulz iterations: 5 | Nesterov: enabled
- Scale correction: `max(1, M/N)^0.5` per matrix

**Adam config** (scalars):
- Betas: (0.9, 0.95) | Eps: 1e-8 | Fused: true

**Why weight decay matters**: WD=0.04 shrinks weight magnitudes, which has two effects:
1. **Better quantization**: Smaller weights have lower quantization error
2. **Better generalization**: Prevents overfitting to training data statistics

**Impact**: +0.003 BPB from WD=0.04 (vs WD=0).

**Verdict**: Standard competition technique. WD=0.04 proven optimal across multiple entries.

---

### 3.7 Stochastic Weight Averaging

**What**: Arithmetic average of model weights collected every 200 steps during the warmdown phase.

```python
if step >= warmdown_start and step % 200 == 0 and step > 100:
    if swa_state is None:
        swa_state = {n: p.clone() for n, p in model.state_dict().items()}
    else:
        for n, p in model.state_dict().items():
            swa_state[n] += p
    swa_count += 1
# After training: swa_state[n] /= swa_count
```

**Why**: During warmdown, the learning rate decays via cosine schedule. SWA averages ~15 checkpoints from this phase, producing smoother weights that quantize better and generalize more.

**Impact**: +0.003 BPB. Smoother weight landscape reduces quantization penalty.

**Verdict**: Free improvement. Replaces EMA (which over-smoothed and hurt by +0.015).

---

### 3.8 Mixed Quantization (int5/int6/int8 + zstd)

**What**: Different precision levels for different model components, compressed with zstd level 22.

| Component | Precision | Range | Size Est. |
|---|---|---|---|
| Embeddings (tok_emb) | int8 | [-128, 127] | ~4 MB |
| Attention weights | int6 | [-32, 31] | ~3.5 MB |
| MLP weights (gate, up, proj) | int5 | [-16, 15] | ~5 MB |
| Biases, RMSNorm | int8 | [-128, 127] | ~0.7 MB |
| Control params (scales, gains) | fp32 | full range | ~0.2 MB |
| Compression overhead (zstd-22) | — | — | ~1.6 MB |
| **Total** | | | **15.02 MB** |

All quantization is per-row with float16 scale factors.

**Why int5 for MLP, int6 for attention**: MLP weights tolerate lower precision (less sensitive to fine-grained values). Attention weights carry positional and contextual information requiring higher fidelity. Embeddings need int8 to preserve vocabulary diversity across 1024 tokens.

**Quantization penalty**:

| Scheme | Size | BPB Penalty |
|---|---|---|
| Pre-quantization (SWA) | 87.4 MB | baseline |
| Int8 all + zlib | 24.7 MB | +0.010 |
| Mixed int5/int6/int8 + zstd-22 | **15.02 MB** | **+0.020** |

The +0.020 penalty is more than offset by LoRA TTT's -0.034 improvement.

**Verdict**: Necessary for 16 MB fit. Int5+int6+int8 is the sweet spot between size and quality.

---

### 3.9 Score-First LoRA TTT

**What**: At evaluation time, inject rank-8 LoRA adapters on Q and V projections across all 11 layers. For each document, process in 256-token chunks: **score** each chunk first (record loss), then **train** LoRA on it so subsequent chunks benefit from adaptation.

```python
class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        self.original = original  # frozen quantized weights
        self.lora_A = Parameter(randn(rank, in_d) * 0.01)
        self.lora_B = Parameter(randn(out_d, rank) * 0.001)
        self.scale = 1.0 / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
```

**Score-then-train loop**:
```
For each document:
    Reset LoRA weights to initial values
    For each 256-token chunk:
        1. SCORE: Forward pass → record NLL and bytes (graded)
        2. TRAIN: Backward pass → Adam step on LoRA params
    → Later chunks benefit from adaptation to earlier context
```

**Why it works**: Documents have a cold-start problem — first tokens average 2.70 loss vs 2.01 for last tokens (0.69 nat gap). LoRA TTT adapts to each document's topic and style, reducing this gap. The rank-8 subspace is expressive enough to capture document-level patterns without overfitting.

**Why score-first is legal**: Every token is graded *before* the model trains on it. This complies with competition rules (confirmed in upstream PR #568). You can't use future tokens to predict past ones.

**LoRA config**:

| Parameter | Value |
|---|---|
| Rank | 8 |
| Targets | Q and V projections (all 11 layers, 22 modules) |
| Learning rate | 0.05 (Adam) |
| Chunk size | 256 tokens |
| Min doc length | 32 tokens (shorter docs scored without TTT) |
| Params per document | ~92K (reset each document) |

**Impact**:

| Approach | val_bpb | Delta |
|---|---|---|
| Quantized model (no TTT) | 1.1930 | — |
| **LoRA TTT** | **1.1573** | **-0.0357** |

**Why SGD TTT fails (+0.018)**: Directly modifying dequantized weights breaks quantization structure. Weights drift out of optimal quantization ranges. LoRA's key advantage: it perturbs in a low-rank subspace while preserving the quantized base weights intact.

**Verdict**: Largest single technique improvement (-0.034). Directly addresses the document cold-start problem that accounts for 36.8% of uncertain-wrong tokens.

---

## 4. Training Progression

### Step-by-Step Trajectory (1xH100 PCIe)

| Step | val_loss | val_bpb | Phase | Δ BPB |
|---|---|---|---|---|
| 0 | 6.9277 | 4.1030 | Init | — |
| 1,000 | 2.2538 | 1.3348 | Rapid descent | -2.768 |
| 2,000 | 2.1475 | 1.2719 | Rapid descent | -0.063 |
| 3,000 | 2.1026 | 1.2453 | Diminishing returns | -0.027 |
| 4,000 | 2.0758 | 1.2294 | Diminishing returns | -0.016 |
| 5,000 | 2.0563 | 1.2179 | Warmdown begins | -0.012 |
| 6,000 | 2.0271 | 1.2006 | Warmdown + SWA | -0.017 |
| 7,000 | 1.9992 | 1.1840 | Warmdown | -0.017 |
| **7,926** | **1.9800** | **1.1727** | **Wallclock cap** | **-0.011** |

### Three-Phase Learning

- **Phase 1 — Rapid Descent (0-2K)**: BPB 4.10→1.27. Rate: -0.065/1K steps. Learning basic token statistics, common words, simple grammar. 20% of steps, 73% of improvement.
- **Phase 2 — Diminishing Returns (2K-5K)**: BPB 1.27→1.22. Rate: -0.010/1K steps. Learning rarer constructions, position-dependent patterns.
- **Phase 3 — Warmdown Acceleration (5K-7.9K)**: BPB 1.22→1.17. Rate: -0.016/1K steps. Cosine LR decay + SWA. **1.6x more efficient than Phase 2**.

### Post-Training Pipeline

| Stage | val_bpb | Delta |
|---|---|---|
| Pre-quant (SWA-averaged) | 1.1727 | — |
| Int8+zlib roundtrip | 1.1826 | +0.010 |
| Mixed quant (int5/int6/int8+zstd) | 1.1930 | +0.020 |
| **+ LoRA TTT** | **1.1573** | **-0.015 net** |

---

## 5. What Each Technique Contributed

| Technique | Est. BPB Gain | How Measured | Critical? |
|---|---|---|---|
| SmearGate | ~0.012 | Ablation: +1.80 loss without | Yes (catastrophic without) |
| seq_len=2048 | +0.015 | Exp 15 vs Exp 14 | Yes |
| 11 layers + U-Net | +0.009 | Exp 14 vs Exp 10 (9L) | Yes |
| XSA (last 4 layers) | +0.005 | Exp 17 vs Exp 15 | Yes |
| SwiGLU MLP | +0.004 | Exp 13 ablation (matched params) | Yes |
| Muon WD=0.04 | +0.003 | Exp 10 ablation | Yes |
| SWA (15 checkpoints) | +0.003 | Pre/post SWA comparison | Yes |
| OrthoInit | ~0.002 | Enables SmearGate; direct ~0.002 | Yes (foundation) |
| Mixed quant penalty | -0.020 | Pre-quant vs post-quant | Necessary |
| **LoRA TTT** | **+0.034** | **Post-quant vs TTT** | **Yes (largest single gain)** |

---

## 6. What Didn't Work

| Technique | Experiment | Result | Why It Failed |
|---|---|---|---|
| Register token | Exp 7-8b | +0.002 worse | Step overhead > marginal juncture benefit |
| Layer looping + wider | Exp 9 | +0.034 worse | Step time from wider dim exceeded benefit |
| Data sampling (juncture) | Exp 11a | +0.002 worse | Shard-level too coarse; data already homogeneous |
| Data sampling (rare bigram) | Exp 11b | +0.011 worse | Reduced diversity hurts overall learning |
| Hard example mining | Exp 11d | +0.040 worse | Fine-tuning destroys Muon weight geometry |
| Partial RoPE (16/64) | Exp 18 | +0.015 worse | 64 head_dim too small to sacrifice 75% of dims |
| EMA (replacing SWA) | Exp 18 | +0.015 worse | Over-smoothed weights; SWA uniform averaging better |
| Heterogeneous MLP | Exp 16 | +0.011 worse | Param reduction > concentration benefit |
| BigramHash | Exp 16 | 0.000 | SmearGate makes it redundant |
| SGD TTT | Exp 17 | +0.018 worse | Modifying dequantized weights breaks quantization |

**Key lesson**: At 26M params with 524K batch, most "clever" techniques don't help. Gains came from proven competition techniques and disciplined hyperparameter tuning.

---

## 7. Bits Budget Analysis

Analysis of 1,024,000 validation tokens:

### Token Difficulty Distribution

| Difficulty | % of Tokens | % of Bits | Avg Loss |
|---|---|---|---|
| Easy (<1 nat) | 44.0% | ~5% | 0.23 |
| Medium (1-3) | 25.4% | ~24% | 1.96 |
| Hard (3-5) | 19.8% | ~37% | 3.90 |
| Very Hard (5+) | 10.8% | ~34% | 6.39 |

The 10.8% very-hard tokens consume 34% of all bits. Hard + very-hard (30.6% of tokens) account for over 70% of total bit budget.

### The Word-Initial Problem

32.5% of hard tokens are word-initial single-character predictions. With a 1024-token vocabulary, `▁s` could start 200+ different words. This is fundamentally a vocabulary problem, not a model capacity problem.

| Context Type | Avg Loss | Ratio to Baseline |
|---|---|---|
| After juncture (`. , the and`) | 3.37 | 1.78x harder |
| Word-initial token | 3.41 | 2.82x harder |
| Word-middle/end | 1.21 | baseline |

SmearGate reduced this gap (without SmearGate: 5.17 avg loss at junctures), but word-initial prediction remains the core challenge.

### Position Utilization

| Position Range | Avg Loss | Notes |
|---|---|---|
| 0-128 | 2.471 | Highest — minimal context |
| 128-256 | 2.121 | Dropping fast |
| 256-512 | 2.066 | Good context |
| 512-1024 | 2.015 | Plateau |
| 1024-2048 | 2.021 | Full context utilized |

Context benefit: 0.69 nats (first 64 tokens at 2.70 vs last 64 at 2.01). Training on seq_len=2048 taught the model to use long-range context, with loss stabilizing around position 512.

### Model Confidence

| Quadrant | % of Tokens |
|---|---|
| Confident Right | 44.2% |
| Uncertain Right | 13.2% |
| Confident Wrong | 5.8% |
| Uncertain Wrong | 36.8% |

Excellent calibration: only 5.8% confident-wrong. The 36.8% uncertain-wrong tokens are at the model's capacity limit — this is what LoRA TTT targets.

---

## 8. Layer & Head Ablations

### Layer Ablation (11L model)

| Layer | Role | BPB Impact if Removed | Observation |
|---|---|---|---|
| **L0** | encoder | **+5.35** | Critical: processes embeddings + SmearGate signal |
| L1 | encoder | +0.92 | Strong secondary encoding |
| L2 | encoder | +0.52 | |
| L3 | encoder | +0.41 | |
| L4 | encoder | +0.24 | |
| L5 | bottleneck | +0.24 | U-Net crossover |
| L6 | decoder | +0.29 | |
| L7 | decoder [XSA] | +0.24 | |
| L8 | decoder [XSA] | +0.19 | Weakest but active |
| L9 | decoder [XSA] | +0.19 | |
| **L10** | decoder [XSA] | **+3.15** | Critical: output prediction, almost entirely MLP |

**Zero dead layers.** Weakest layer (L8) at 0.19 still meaningfully contributes. Compare to 9L model where layers 4-6 were near-dead at 0.10-0.14.

### Top Attention Heads

| Head | Impact | Notes |
|---|---|---|
| L0H6 | +0.775 | Primary first-layer head |
| L0H0 | +0.583 | Second critical first-layer head |
| L1H7 | +0.118 | Important second-layer head |
| L7H7 | +0.082 | XSA layer head |
| L6H6 | +0.078 | |

All 88 heads contribute meaningfully — no dead heads.

---

## 9. Experiment Journey

18 experiments, 5 days, ~$50 total compute:

| Exp | Day | val_bpb | What Changed | Key Finding |
|---|---|---|---|---|
| 1 | 1 | 3.10 | Wrong batch size (256) | Batch size is critical |
| 2 | 1 | 1.46 | Fixed batch to 65K | Correct batching baseline |
| 6 | 1 | 1.312 | 1200s training, warmdown=600 | Longer warmdown better |
| 8a | 2 | 1.299 | 9L baseline, warmdown=3000 | warmdown=3000 optimal |
| **10** | **2** | **1.283** | **+ SmearGate, OrthoInit, MLP 3x, WD** | **Competition stack works** |
| 11a-d | 3 | — | Data sampling variants | Dead end |
| 13 | 3 | — | SwiGLU > ReLU² | +0.004, activates L10 |
| **14** | **3** | **1.278** | **+ 11 layers + SWA** | **No dead layers** |
| 14-8x | 4 | 1.202 | Same on 8xH100 | Batch size: +0.076 BPB |
| **15** | **4** | **1.187** | **+ seq_len=2048** | **+0.015, context benefit** |
| **17** | **4** | **1.183** | **+ XSA (last 4 layers)** | **+0.005, consistent** |
| Quant | — | 1.191 | int5+int6+int8+zstd | 15 MB target achieved |
| **+ TTT** | **5** | **1.157** | **LoRA TTT** | **-0.034 via test-time adapt** |

### Key Decision Points

1. **Exp 2**: Fixed batch size 256→65K = 1.64 BPP improvement. Most impactful single fix.
2. **Exp 10**: Validated that published competition techniques (SmearGate, OrthoInit, MLP3x, WD) work at this scale.
3. **Exp 11a-d**: Data sampling experiments all failed. Proved architecture/batch is the bottleneck, not data selectivity.
4. **Exp 14-8x**: 8xH100 run showed batch size dominates: 0.076 of 0.097 total improvement came from 8x batch.
5. **LoRA TTT**: Score-first strategy enabled -0.034 BPP, directly targeting cold-start problem.

---

## 10. Comparison to Competition

### Leaderboard Context (as of 2026-03-24)

| Rank | Entry | BPB | Gap from Us |
|---|---|---|---|
| #1 | LeakyReLU² + Legal TTT + Parallel Muon | 1.1194 | 0.038 ahead |
| #2 | 11L EMA + GPTQ-lite + warmdown3500 | 1.1228 | 0.035 ahead |
| #9 | SmearGate + OrthoInit + Muon WD | 1.1556 | 0.002 ahead |
| **Ours** | **11L XSA + Score-First LoRA TTT** | **1.1573** | **—** |
| Baseline | Naive 9L 512d | 1.2244 | 0.067 behind |

### Estimated Gap Breakdown (0.038 to #1)

| Source | Est. BPB |
|---|---|
| FA3 (Flash Attention 3) kernel | ~0.005 |
| Better LR schedule (0.025 + momentum) | ~0.005 |
| Sliding window eval (stride=64) | ~0.005 |
| QAT (late-stage quantization-aware training) | ~0.005 |
| Parallel Muon optimizer | ~0.005 |
| 3-seed averaging + tuning | ~0.005 |
| Remaining unknown | ~0.008 |

---

## 11. Conclusion

This submission demonstrates that systematic combination of proven techniques yields competitive results at the 16 MB constraint:

- **Architecture**: 11L U-Net transformer with SmearGate, SwiGLU, XSA, OrthoInit — zero dead layers, all 88 heads active
- **Quantization**: Mixed int5/int6/int8 + zstd-22 achieves 15.02 MB with +0.020 BPB penalty
- **Test-time adaptation**: Score-first LoRA TTT provides -0.034 BPB, the largest single technique contribution
- **Net result**: 1.1573 BPB, 0.067 better than baseline, trained for ~$50 total

The dominant bottleneck remains word-initial prediction (32.5% of hard tokens) driven by the 1024-token vocabulary, and batch size effects (0.076 BPB gap between 1x and 8x H100). An 8xH100 record run with FA3, QAT, and optimized LR schedule could close most of the 0.038 gap to the current leader.
