# Parameter Golf: Definitive Technique Reference

*Deep code-level analysis of every recurring method. Extracted from actual submission source code.*

---

## Table of Contents

1. [Quantization Pipeline](#1-quantization-pipeline)
2. [Architecture Components](#2-architecture-components)
3. [Training Infrastructure](#3-training-infrastructure)
4. [Evaluation Pipeline](#4-evaluation-pipeline)

---

## 1. Quantization Pipeline

Quantization is the single most impactful technique family — it determines how many parameters fit in 16MB.

### 1.1 Int6 STE QAT (Quantization-Aware Training)

**What:** During training, weights are fake-quantized to 6-bit precision (64 levels, range [-32, 31]) every forward pass. The Straight-Through Estimator (STE) passes gradients through the rounding as if it didn't happen.

**Code (from CastedLinear.forward):**
```python
with torch.no_grad():
    w32 = self.weight.float()
    row_max = w32.abs().amax(dim=1)                          # per-row max
    scale = (row_max / 31.0).clamp_min(1.0 / 31.0)          # scale: maps max → 31
    w_q = (torch.clamp(torch.round(w32 / scale[:, None]),
                        -32, 31) * scale[:, None]).to(x.dtype)
w = w + (w_q - w).detach()    # STE: forward uses w_q, backward uses w
```

**Math:** Per row: `scale = max(|w|) / 31`. Each weight: `q = clamp(round(w / scale), -32, 31)`. Dequantized: `w_q = q * scale`. The STE trick: `w + (w_q - w).detach()` equals `w_q` in the forward pass, but `d(loss)/dw` flows through as if `w` was used directly.

**When enabled:** Late QAT — only when LR scale drops below 0.15 (last ~15% of training). This lets the model train in full precision most of the time, then fine-tune with the quantization grid active.

**Impact:** Near-zero quantization gap (pre-quant vs post-quant BPB difference drops from ~0.007 to ~0.0001).

---

### 1.2 GPTQ-lite (Percentile Clip Search)

**What:** Post-training quantization that tries 5 different clipping thresholds per weight row and picks whichever minimizes reconstruction MSE.

**Code:**
```python
for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
    if pct < 1.0:
        row_clip = torch.quantile(t32.abs(), pct, dim=1)
    else:
        row_clip = t32.abs().amax(dim=1)
    s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()[:, None]),
                     -clip_range, clip_range).to(torch.int8)
    recon = q.float() * s.float()[:, None]
    err = (t32 - recon).pow(2).mean().item()
    if err < best_err:
        best_q, best_s, best_err = q, s, err
```

**Why:** Weight rows often have a few outliers that inflate the scale. Clipping at the 99.9th percentile sacrifices those outliers but gives much tighter quantization to the remaining 99.9% of weights. The 5-candidate search finds the best tradeoff automatically.

**Impact:** -0.0006 BPB over naive abs-max scaling. Zero training cost — runs once after training.

---

### 1.3 Full Hessian GPTQ

**What:** Uses the Hessian matrix H = X^TX (capturing how sensitive the loss is to each weight) to compensate quantization errors across columns.

**Key steps:**
1. **Collect Hessians:** Register forward hooks on every Linear layer. For each batch of calibration data, accumulate `H += X^T @ X` where X is the layer input.
2. **Column reordering:** Sort columns by descending `H[i,i]` (most sensitive first).
3. **Cholesky of H^{-1}:** Compute `L = cholesky(H)`, then `H^{-1} = cholesky_inverse(L)`, then `U = cholesky(H^{-1}, upper=True)`.
4. **Column-by-column quantization:** For each column i, quantize it, compute error `e = (w - q*s) / U[i,i]`, then propagate: `W[:, i+1:] -= e * U[i, i+1:]`. This adjusts remaining columns to compensate for the error just introduced.
5. **Block processing:** Process 128 columns at a time with inter-block propagation.

**Why it's better than GPTQ-lite:** GPTQ-lite treats each row independently. Full GPTQ treats columns jointly — when one column is poorly quantized, neighboring columns are adjusted to compensate, minimizing the total reconstruction error across the entire weight matrix.

**Impact:** -0.002 to -0.005 BPB over GPTQ-lite.

---

### 1.4 AR Self-Generated Calibration

**What:** The model generates its own calibration data for GPTQ Hessian collection, avoiding any dependency on training or validation data.

**Code:**
```python
def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, seed=42):
    tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
    for pos in range(seq_len - 1):
        logits = model.forward_logits(tokens)
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_tok = torch.multinomial(probs, 1, generator=rng)
        tokens = torch.cat([tokens, next_tok], dim=1)
```

**Why:** Prior GPTQ implementations calibrated on training data, which the rules prohibit accessing after the 600s training window. AR self-generation is fully legal — the model samples from its own distribution. It closes 84% of the quality gap vs. using actual validation data for calibration (only +0.0003 BPB penalty).

---

### 1.5 Int5 Quantization (for MLP weights)

**What:** 5-bit quantization (32 levels, range [-16, 15]) applied specifically to MLP weights, while attention weights stay at int6.

**Why int5 for MLP, int6 for attention:** MLP weights are ~4x more numerous (hidden=1536 vs dim=512) but less sensitive to precision loss. Attention weights (Q/K/V/Out projections) directly shape attention patterns and are more precision-critical. Using int5 on MLP saves enough bytes to fund an extra transformer layer.

**Impact:** -0.003 BPB (from the extra layer funded by the savings).

---

### 1.6 Selective ±1 Pruning

**What:** After GPTQ, find all weights quantized to ±1, compute each one's reconstruction error cost (scale²), sort ascending, and zero the cheapest ones until the compressed artifact fits the target size.

**Why not magnitude pruning:** This is more surgical — it specifically targets the quantized values where zeroing has the least impact on output quality, rather than just zeroing small FP32 weights.

---

### 1.7 Compression Stack

| Method | Used by | Ratio | Notes |
|--------|---------|-------|-------|
| zlib-9 | Early submissions | Baseline | |
| zstd-22 | PR#198-PR#414 | ~1.5MB saved vs zlib | Better for int6 data |
| LZMA preset=9 | PR#1019 (SOTA) | Slightly better than zstd | Slower compression, but only runs once |
| Brotli-11 | Frontier PRs | Comparable to LZMA | Some frontier PRs prefer it |

Int6 values stored in int8 containers have restricted entropy (63 of 256 possible values), so compression is very effective. Selective ±1 pruning increases the zero proportion, further improving compression ratios.

---

### 1.8 Quantization Evolution Summary

```
Baseline:     int8 + zlib             → ~0.007 BPB quant gap
  ↓
+ STE QAT:    int6 + zstd             → ~0.0001 BPB quant gap
  ↓
+ GPTQ-lite:  5 clip percentiles      → -0.0006 BPB
  ↓
+ Full GPTQ:  Hessian + Cholesky      → -0.003 BPB
  ↓
+ AR self-gen: legal calibration       → +0.0003 BPB penalty (acceptable)
  ↓
+ ±1 pruning: surgical size control    → fits exactly in 15.9MB
```

---

## 2. Architecture Components

### 2.1 SmearGate

**What:** Per-dimension learned gate blending each token's embedding with the previous token's.

```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Parameter(torch.zeros(dim))  # init: sigmoid(0)=0.5

    def forward(self, x):
        g = torch.sigmoid(self.gate)[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```

**Why:** Injects bigram context into embeddings for free (~512 params). The transformer doesn't need to "rediscover" that token pairs co-occur — SmearGate hands it that signal directly.

**Caveat:** PR#363 found SmearGate **hurts** depth-recurrent architectures. If using layer looping, consider removing it.

---

### 2.2 BigramHash

**What:** Hash function maps (prev_token, cur_token) pairs to embedding buckets. Adds bigram statistics as an additive signal to token embeddings.

```python
# Hash: XOR-based, avoids collisions better than linear
hash_idx = (36313 * cur ^ 27191 * prev) % (num_buckets - 1)
bigram_emb = self.embed(hash_idx)       # [num_buckets, 128]
projected = self.proj(bigram_emb)       # [128] → [512]
return projected * self.scale           # scale init=0.05
```

**Scaling:** 2048 buckets (early) → 4096 → 10240 → 3072×112 (SOTA, smaller dim to fit budget).

---

### 2.3 XSA (Exclusive Self-Attention)

**What:** After standard attention, subtract the component of the output aligned with each position's own value vector.

```python
def _xsa_efficient(self, y, v):
    # y: [B,T,H,D], v: [B,T,Hkv,D]
    y_g = y.reshape(B, T, Hkv, group, D)          # group query heads by KV head
    vn = F.normalize(v, dim=-1).unsqueeze(-2)      # [B,T,Hkv,1,D]
    proj = (y_g * vn).sum(-1, keepdim=True) * vn   # self-aligned component
    return (y_g - proj).reshape(B, T, H, D)        # subtract it
```

**Math:** `y_xsa = y - (y · v̂) v̂` where `v̂ = v/||v||`. This is a vector rejection — removes the self-component, keeping only cross-position information.

**Why the efficient version matters:** Standard implementation needs `v.repeat_interleave(group_size)` which allocates a new tensor. The reshape+broadcast version does the same math with zero extra memory.

**Impact:** -0.002 BPB (partial, last 3-4 layers) or -0.006 BPB (all 11 layers).

---

### 2.4 Partial RoPE

**What:** Only 16 of 64 head dimensions get rotary positional embeddings. The other 48 are position-free.

```python
x_rope, x_pass = x[..., :16], x[..., 16:]
x_rope = apply_rotary(x_rope, cos, sin)
return torch.cat([x_rope, x_pass], dim=-1)
```

**Why:** Most of the attention can be pure content-matching without positional bias. Only a small fraction needs position information. This slightly improves performance, especially for sequences shorter than the RoPE frequency range.

---

### 2.5 LN Scale

**What:** After RMSNorm, multiply by `1/sqrt(layer_idx + 1)`. Zero parameters.

```python
self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1)
# Layer 0: 1.0, Layer 5: 0.408, Layer 10: 0.302
```

**Why:** Dampens deeper layers' activation magnitudes. The residual stream grows as ~sqrt(depth); this normalizes that growth, stabilizing training in deep models.

---

### 2.6 VE128 (Shared Value Embeddings)

**What:** A shared 128-dim embedding table looked up by token ID, projected to KV dim, and added to attention values at layers 9-10 only.

**Why layers 9-10 only:** Last layers need token identity for prediction. Deep residual transformations may have diluted this signal. VE128 reinjects it cheaply.

---

### 2.7 U-Net Skip Connections

**What:** 11 layers split into 5 encoder + 6 decoder. Encoder activations skip-connect to decoder layers (layer 4→5, 3→6, 2→7, 1→8, 0→9).

```python
# Encoder phase: save activations
skips.append(x)
# Decoder phase: add skip with learned per-dimension scale
x = x + skip_weights[i][None, None, :] * skips.pop()
```

**Why:** Provides information shortcuts from early (local/token-specific) to late (abstract/prediction) layers.

---

### 2.8 LeakyReLU(0.5)²

**One-line change from ReLU²:**
```python
# ReLU²:      x = torch.relu(up(x)).square()
# LeakyReLU²: x = F.leaky_relu(up(x), 0.5).square()
```

**Effect:** Negative pre-activations produce `(0.5x)² = 0.25x²` instead of 0. No dead neurons. -0.002 to -0.003 BPB.

---

### 2.9 Depth Recurrence (Frontier Technique)

**What:** Share layer weights and loop through them 2-3 times, creating more effective depth without more parameters.

```
Physical: 9 layers with unique weights
Effective: 18 layers (each physical layer runs twice)
Parameter cost: same as 9 layers
```

**Impact:** -0.01 to -0.02 BPB. Dominant technique in the 1.085-1.09 frontier.

**Key caveat:** SmearGate hurts looped architectures (PR#363). The frontier PRs that use depth recurrence typically remove SmearGate.

---

## 3. Training Infrastructure

### 3.1 Muon Optimizer

**What:** Replaces Adam's per-element scaling with Newton-Schulz orthogonalization of the gradient matrix.

**Core: Newton-Schulz iteration (5 steps):**
```python
a, b, c = (3.4445, -4.7750, 2.0315)  # quintic polynomial coefficients
X = G / G.norm()                       # normalize
for _ in range(5):
    A = X @ X.T
    B = b*A + c*(A@A)
    X = a*X + B@X                      # converges to orthogonal factor
```

**Configuration:**
- Momentum: warmup from 0.92 → 0.99 over 1500 steps
- Weight decay: 0.04 (decoupled: `p *= 1 - lr * wd` before update)
- Applied to all 2D weight matrices (via Parameter Banking)

---

### 3.2 Parameter Banking

**What:** 66 separate Linear weights restructured into 4 contiguous 3D tensors:

```python
qo_bank     = Parameter([22, 512, 512])    # Q + Out projections
kv_bank     = Parameter([22, 256, 512])    # K + V projections
mlp_up_bank = Parameter([11, 1536, 512])   # MLP up projections
mlp_down_bank = Parameter([11, 512, 1536]) # MLP down projections
```

**Why:** Newton-Schulz becomes a single batched `bmm` over the batch dimension. 19.7ms → 1.3ms (15x faster). Also reduces DDP communication from 66 small ops to 4 large ops.

---

### 3.3 EMA + Tight SWA

**EMA (every step):**
```python
ema[name] = 0.997 * ema[name] + 0.003 * param  # lookback ~333 steps
```

**Tight SWA (only last 20% of warmdown, every 50 steps):**
```python
if lr_scale < 0.2 and step % 50 == 0:
    swa_state[name] += param; swa_count += 1
```

EMA is the default final weight source. Tight SWA captures the center of the final loss basin.

**Key negative finding:** SWA sabotages QAT (PR#989, -3.64 mBPB). Don't combine them naively.

---

## 4. Evaluation Pipeline

### 4.1 Sliding Window Eval (stride=64)

**What:** Instead of scoring all tokens with 0-to-N context, advance a window by `stride=64` tokens at a time. Each scored token gets ~960 tokens of context.

```
Window 0: tokens [0, 1024)    → score all 1024 (first window special)
Window 1: tokens [64, 1088)   → score only [960, 1024) of window = tokens [1024, 1088)
Window 2: tokens [128, 1152)  → score only [960, 1024) = tokens [1088, 1152)
...
```

**Impact:** -0.032 BPB. FREE — zero training change, zero artifact cost. 16x more forward passes at eval time (~120s vs ~7s).

---

### 4.2 Legal Score-First TTT

**Protocol:**
1. Split validation tokens into 32K-token chunks
2. For each chunk:
   - **SCORE** under `torch.inference_mode()` (stateless, no gradient tracking)
   - **TRAIN** with SGD(lr=0.002, momentum=0.9), 3 epochs, cosine LR decay across chunks
3. Last chunk: score only, never train

**Why SGD not AdamW:** Adam's second moment needs many steps to warm up. SGD+momentum carries directional information across chunks without per-element state overhead.

**Timing:** ~410s for TTT. Total budget: 600s train + ~120s eval + ~410s TTT ≈ 19 min (within 20 min combined limit).

**Key finding (PR#1341):** TTT and GPTQ are fundamentally incompatible. Full Hessian GPTQ captures much of what TTT provides, making TTT neutral or negative on GPTQ-optimized models.

---

### 4.3 Full End-to-End Pipeline

```
Train (600s, Muon+AdamW, late QAT)
  → Apply EMA weights
  → Unbank 3D tensors → 2D matrices
  → AR self-generate 64×2048 calibration tokens
  → Collect Hessians (forward hooks, H = X^TX)
  → Full Hessian GPTQ (int6, per-row, 5 clips)
  → Selective ±1 pruning (binary search to fit budget)
  → LZMA compress → write artifact
  → Decompress → dequantize → rebank → compile
  → Sliding window eval (stride=64) → BPB
  → (Optional) Legal TTT → TTT BPB
```

---

*Source files: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`, `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/train_gpt_v5.py`, `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py`*
