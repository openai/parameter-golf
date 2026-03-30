# Competition Intel Deep Dive (2026-03-23)

## Purpose

Code-level analysis of top non-TTT entries to understand the exact architectural differences
driving their scores, identify implementation divergences from our stack, and find free lunches.

---

## PR #505 (1.1181 BPB) — Best Non-TTT Entry

**Author:** JoeProAI | **Artifact:** 15.71 MB | **Params:** ~28.0M | **3-seed mean:** 1.11808

### Architecture (exact code)

**Full MHA, not GQA:**
```python
num_heads = 8, num_kv_heads = 8  # NOT 4
kv_dim = 8 * 64 = 512           # same as model_dim
# K/V projections are full-rank: CastedLinear(512, 512)
```
This is the single biggest architectural difference from us. Their attention has 2x the K/V capacity.

**Star-ReLU MLP (NOT SwiGLU despite PR title):**
```python
class MLP:
    up_proj = CastedLinear(512, 1792, bias=False)
    down_proj = CastedLinear(1792, 512, bias=False)  # zero-init
    scale = nn.Parameter(torch.ones(1792))   # per-hidden learned scale
    bias = nn.Parameter(torch.zeros(1792))   # per-hidden learned bias

    def forward(x):
        return down_proj(relu(up_proj(x))**2 * scale + bias)
```
Single up-projection, no gate. Our Star-ReLU matches this pattern.

**Sigmoid Skip Gates:**
```python
skip_weights = nn.Parameter(torch.ones(5, 512))    # init 1.0
skip_gates = nn.Parameter(torch.zeros(5, 512))     # init 0.0 → sigmoid = 0.5

# In decoder:
gate = sigmoid(skip_gates[i])                       # per-dim, shape [512]
x = gate * x + (1 - gate) * (skip_weights[i] * encoder_skip)
```
Starts at 50/50 blend. Per-dimension (512 gates per skip connection).

**VE128 injection (before head reshape):**
```python
v = self.c_v(x).reshape(bsz, seqlen, num_kv_heads, head_dim)  # WAIT — reshape first
# Then VE added:
ve_reshaped = v_embed.reshape(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)
v = v + ve_reshaped  # added AFTER reshape to heads
```
VE projects to kv_dim=512 (because MHA), then reshaped to match v's head layout.

**Decoder 2x LR:**
```python
# Encoder matrix params → Muon lr=0.025
# Decoder matrix params → Muon lr=0.025 * 2.0 = 0.05
# Encoder scalar params → AdamW lr=0.025
# Decoder scalar params → AdamW lr=0.025 * 2.0 = 0.05
```

**Late QAT (torch.compile compatible):**
```python
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False  # CLASS-LEVEL flag, not instance

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                scale = w.float().abs().amax(dim=1) / 31.0
                w_q = (clamp(round(w / scale), -32, 31) * scale).to(x.dtype)
            w = w + (w_q - w).detach()  # STE
        return F.linear(x, w, bias)

# Activated in training loop:
if scale < 0.15:  # last 15% of warmdown
    CastedLinear._qat_enabled = True
```
Works with torch.compile because class attributes are read dynamically, not constant-folded.

**Quantization (how 1792 fits in 16MB):**
```python
# Int6 per-row: all 2D matrices with numel > 65536 (except tok_emb)
# fp16: tok_emb.weight + all tensors with numel <= 65536
# fp32: control tensors (attn_scale, mlp_scale, skip_weights, skip_gates, VE scales)
# Compression: zstd level 22
# Per-row clipping: 99.99984th percentile before quantization
```
A 512×1792 matrix at int6 = ~918KB vs ~1.8MB at fp16. That's how they fit.

**EMA (not SWA):**
```python
# Every step: ema_state[name] = 0.997 * ema_state[name] + 0.003 * param
# Applied BEFORE quantization. SWA is dead code when EMA is on.
```

**Training config:**
- batch_tokens=786,432, seq_len=2048, warmdown=3500
- MATRIX_LR=0.025, Muon momentum 0.92→0.99 over 1500 steps
- No grad clipping (grad_clip_norm=0.0)
- ~48ms/step on their hardware → ~12,500 steps in 600s

### CRITICAL: PR #505 Does NOT Fit in 16MB

A commenter confirmed they got a ~20MB artifact when running #505's code. The reported
15.71MB was likely a pre-quant or subset measurement. At 8 KV heads + h=1792 + BigramHash 8192,
the model is ~33.5M params → ~20MB under int6+zstd-22. **This PR cannot be submitted as-is.**

This changes the competitive picture significantly: the actual best non-TTT score that fits
in 16MB may be lower. PR #549 (1.1194 with legal TTT) and PR #445 (1.1236 no TTT) are the
real targets.

### Can We Fit 8 KV Heads?

| Config | Params | Est. Artifact | Fits? |
|--------|--------|---------------|-------|
| Our model (4 KV) | ~27.5M | ~16.0MB | Barely |
| Our model (8 KV) | ~30.4M | ~17.8MB | **No** |
| #505 (8KV, h=1792) | ~33.5M | ~20.0MB | **No** |

Going to 8 KV heads adds ~2.9M params / ~1.7MB. Not viable without cutting something else.

### Revised Gap Analysis

The real non-TTT target is **#445 at 1.1236** (Late Training Replay). Our gap is 0.026 BPB.

| Factor | Us | #445 | Impact |
|--------|-----|------|--------|
| Late Training Replay | No | Yes (100 batch, 2ep) | -0.01 to -0.02 |
| Late QAT | No | Yes (<0.15) | -0.002 to -0.005 |
| MATRIX_LR | 0.025 | 0.025 | Both same (but 0.03 may help) |
| VE128 | Yes | Yes | Same |
| Our unique: VR, GA, per-layer LR, GradQuant, Trigram | Yes | No | Our edge |

---

## PR #374 (1.1246 BPB) — Tight SWA + VE128

**Author:** vadim borisov | **Artifact:** 15.71 MB

### Tight SWA (the innovation)
```python
# Collection trigger: scale < 0.2 (last ~20% of warmdown)
# vs standard: scale < 0.5 (last ~50%)
# Frequency: every 50 steps
# Result: ~12 checkpoints from the tail, not ~30 from the middle
# Post-SWA BPB = pre-SWA BPB (zero penalty)
```
Key insight: standard SWA averages stale checkpoints from early warmdown. Tight SWA only averages the final converged region.

### VE128 implementation
```python
class ValueEmbedding:
    embed = nn.Embedding(1024, 128)          # normal init std=0.01
    proj = CastedLinear(128, kv_dim=256)     # zero-init (kv_dim=256 because GQA 4 heads)
    scale = nn.Parameter(tensor(0.1))        # global scale

# Per-layer: nn.Parameter(ones(1)) for each of layers 9, 10
# Cached: computed once per forward, scaled per layer
```

### Block.forward
```python
def forward(x, x0, v_embed=None):
    x_in = resid_mix[0] * x + resid_mix[1] * x0   # learned blend with x0
    attn_out = attn(attn_norm(x_in) * ln_scale_factor, v_embed=v_embed)
    x = x_in + attn_scale * attn_out
    x = x + mlp_scale * mlp(mlp_norm(x) * ln_scale_factor)
    return x
```
Same pattern as ours. `ln_scale_factor = 1/sqrt(layer_idx+1)`.

---

## PR #445 (1.1236 BPB) — Late Training Replay

**Base:** Same as #374 + EMA

### Late Training Replay (exact implementation)
```python
# Config:
ttt_burst_epochs = 2
ttt_burst_lr_factor = 0.1    # 10% of base LR
ttt_burst_steps = 100         # buffer last 100 training batches

# Buffer construction (during warmdown, scale < 0.2):
train_loader._ttt_buffer.append((x.detach().clone(), y.detach().clone()))
# Rolling window: keeps last 100 batches (FIFO)

# Replay loop (AFTER main training, BEFORE EMA application):
for epoch in range(2):
    for (bx, by) in buffer:
        # Set LR to base_lr * 0.1 for ALL optimizer groups
        loss = model(bx, by)
        loss.backward()
        # Grad clipping still active
        for opt in optimizers:  # Muon + AdamW
            opt.step()
        # EMA updated during replay too!
        ema_state[name].mul_(0.997).add_(param, alpha=0.003)
```

**Critical detail:** EMA is updated during the replay phase. The "sharpened" signal from replay propagates into EMA weights. This is 200 extra optimizer steps at 10% LR.

**Impact vs base:** ~0.001 BPB improvement (marginal in their testing).

---

## PR #486 (1.1101 BPB) — Value Residual + TrigramHash + GradQuant

**Uses TTT (10 epochs AdamW)** — score not directly comparable for non-TTT path.

### Value Residual (exact code)
```python
# Init:
self.vr_lambda = nn.Parameter(tensor([0.5, 0.5]))  # 2 params per layer

# Forward:
v = c_v(x).reshape(bsz, seqlen, num_kv_heads, head_dim)
raw_v = v   # returned for caching
if v0 is not None:
    v = lambda[0] * v0 + lambda[1] * v  # blend with layer-0's V
# First layer: v0 is None, so raw_v is cached as v0 for all subsequent layers
```
**Operates on reshaped V (after head split)** — different from VE128 which operates before reshape.

### TrigramHash (exact code)
```python
# Hash: xor(36313*t[i], 27191*t[i-1], 51497*t[i-2]) % 4095
# Embed: nn.Embedding(4096, 128), init zeros
# Proj: CastedLinear(128, 512), init zeros
# Scale: nn.Parameter(tensor(0.03))  # smaller than bigram's 0.05
```

### GradQuant (exact code)
```python
# Accumulation: during scale < 0.1 (last 10% of warmdown)
# Per tensor: mean(grad**2) accumulated over steps
# Bit allocation:
#   Top 10% sensitivity → Int7 (63 levels)
#   Middle 70% → Int6 (31 levels)
#   Bottom 20% → Int5 (15 levels)
# Only applies to 2D tensors with >=64 rows
# Runs on raw gradients BEFORE grad clipping
```

---

## New Competition Developments

### Verified Free Lunch: MATRIX_LR=0.03

PR #530 found via systematic sweep that **MATRIX_LR=0.03 improves by 0.059 BPB** over the
default 0.02 (and likely ~0.005-0.01 over our 0.025). Simple one-line change, no risk.

### LeakyReLU(0.5)^2

Now standard in top entries (#518, #549, #535). Formula: `max(x, 0.5*x)^2`. Different from
Star-ReLU which is `relu(x)^2 * scale + bias`. May be worth testing as a swap.

### DG Attention (PR #542, 1.1898 BPB)

Novel: deep layers transmit "what's new" instead of raw content. Learned per-layer beta
controls the blend. Beta trajectory: 0.758 (layer 0, mostly raw) → 0.401 (layer 10, mostly
differential). Step time too slow (201ms) but conceptually interesting for architecture innovation.

### BitNet b1.58 + Depth Recurrence (PR #540, Draft)

Ternary weights {-1, 0, +1} with LSQ scale. 4 unique blocks × 6 loops = 24 effective layers
(48 at eval). 61.9M params in ~14.2MB. No results yet but the math works for 4x more effective
parameters in 16MB.

### FP8 Training + Arithmetic Coding (PR #538, 1.1511 BPB)

TransformerEngine FP8 for 1.3-1.5x throughput. Custom arithmetic coder replacing zstd-22,
exploiting peaked weight distributions. Per-tensor empirical histograms approach Shannon entropy.

---

## Implementation Divergences: Us vs Leaders

### Confirmed Matching
- Star-ReLU activation (matches #505)
- VE128 on layers 9,10 (matches #374)
- Value Residual lambda init [0.5, 0.5] (matches #486)
- TrigramHash function and init (matches #486)
- GradQuant sensitivity accumulation before grad clip (matches #486)
- Warmdown schedule (wall-clock based, matches all)

### Confirmed Divergent
- **KV heads: 4 vs #505's 8** — biggest architectural gap
- **MLP hidden: 1536 vs #505's 1792** — can't fit in 16MB at our quant level
- **MATRIX_LR: 0.025 vs optimal 0.03** — free lunch not yet adopted
- **Batch tokens: 524K vs #505's 786K** — they run at 48ms/step, we can't
- **Grad clip: 0.3 (run3) vs 0.0 (#505)** — run3 used 0.3, need to test 0.0
- **Decoder 2x LR: off (run3) vs on (#505)** — run6 showed it hurts us, may only work with MHA
- **Late QAT: off vs on (#505/#374/#445)** — run5 showed wash, need isolated test
- **Late Training Replay: missing** — #445's unique technique, 200 extra steps at 10% LR

### Unknown / To Verify
- Our VE injection point — need to confirm it's before head reshape (matching #374)
- Our SWA trigger — should be scale < 0.2 (tight), verify not wider
- Bigram hash bucket count — run3 used 4096, some leaders use 2048 or 8192

---

---

## Novel Technique Deep Dives

### DG Attention (PR #542, 1.1898 BPB)

**Core idea:** Deep layers transmit "what's new" (differential signal) instead of raw content.

```python
# Per layer: fixed linear schedule
raw_mix = 1.0 - (layer_idx / (num_layers - 1))
# L0 = 1.0 (pure raw), L10 = 0.0 (pure differential)

# Novelty = current token - previous token (projected)
prev = F.pad(x[:, :-1], (0, 0, 1, 0))      # shift right, pad zero
diff_signal = c_payload(x) - c_payload(prev)  # "what's new"
payload = raw_mix * c_payload(x) + (1 - raw_mix) * diff_signal
# Standard attention on (designator_q, designator_k, payload)
```

**Learned beta trajectory** shows model naturally uses the schedule:
- L0: 0.758 (mostly raw content)
- L10: 0.401 (mostly differential/novelty)

**Bottleneck:** 201ms/step from double `c_payload` projection (once for x, once for prev)
on 9/11 layers. Could be halved by batching both projections into one matmul.

**Innovation potential for us:** The differential encoding concept could be added as a
lightweight modification to our attention — a learnable `raw_mix` scalar per layer that
blends standard V with `V - V_prev`. No extra projections needed if we operate on the
V tensor directly. Cost: 11 scalar params + one tensor shift per layer.

### BitNet b1.58 + Depth Recurrence (PR #540, Draft)

**Core idea:** Ternary weights {-1, 0, +1} with learned scale. 4 unique blocks × 6 loops
= 24 effective layers (48 at eval). 61.9M params compressed to ~14.2MB.

```python
# LSQ (Learned Step Quantization):
alpha = exp(log_alpha)                    # learned per-layer scale
w_norm = weight / alpha
w_q = clamp(round(w_norm), -1, 1)        # ternary
w_ternary = (w_norm + (w_q - w_norm).detach()) * alpha  # STE
```

**Why it avoids Huginn's failure (4.34 BPB):**
The `resid_mix` anchor to `x0` (initial embedding) prevents representation drift across loops.
Per-dimension learned blend between running state and anchor. This is the critical difference
from naive depth recurrence.

**Eval-time scaling:** 6 loops (train) → 12 loops (eval) = free extra compute. The x0 anchor
keeps representations stable even at 2x the training depth.

**No results yet.** If it works, this is the highest-leverage technique in the competition:
4x more effective parameters in the same 16MB.

### Arithmetic Coding (PR #538, replaces zstd-22)

**Core idea:** Per-tensor empirical histogram as probability model → integer arithmetic coding
→ approaches Shannon entropy limit. Generic compressors like zstd waste bits on the peaked
distributions of quantized neural network weights.

```python
# Per tensor: build CDF from empirical histogram
hist = bincount(quantized_weights)
cdf = cumsum(hist) * 65536 / total         # 16-bit precision

# Arithmetic encode each symbol using CDF
# Standard 32-bit interval arithmetic with renormalization
```

**Decoupled from FP8** — can be adopted as a drop-in replacement for zstd-22 in our
quantization pipeline. Only dependency: numpy + struct + multiprocessing.

**Potential:** Could save 10-20% artifact size vs zstd-22 on peaked int5/int6 distributions.
This could be the key to fitting 8 KV heads or wider MLP in 16MB.

### LeakyReLU(0.5)^2 (PR #518/549)

```python
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

Zero extra params. Ablated at **-0.003 BPB** vs ReLU^2. Addresses dead neuron problem by
letting negative pre-activations flow through (attenuated at 0.5x, then squared). Our
Star-ReLU uses `relu(x)^2 * scale + bias` which keeps dead neurons but adds learned affine.

**Direct comparison:**
- LeakyReLU(0.5)^2: 0 extra params, -0.003 BPB proven
- Star-ReLU: 3584 extra params/layer, unproven advantage over LeakyReLU

Worth testing as a swap.

### MATRIX_LR=0.03 (PR #530)

Systematic sweep found **0.03 beats 0.025 by ~0.005-0.01 BPB**. One-line change, verified,
transferable across architectures. Free lunch.

---

## Priority Actions (Revised)

### Tier 1: Free Lunches (config only, no risk)
1. **MATRIX_LR=0.03** — verified -0.005+ BPB, one line change
2. **LeakyReLU(0.5)^2** — zero params, -0.003 BPB proven, one line swap

### Tier 2: High-Value Code Changes
3. **Late Training Replay** — 200 extra steps at 10% LR, EMA-aware, fully legal (~50 lines)
4. **Arithmetic Coding** — replace zstd-22, could save 10-20% artifact size (~200 lines)
5. **Lightweight DG (differential V)** — add learnable raw_mix per layer, blend V with V_shifted (~20 lines)

### Tier 3: Our Novel Innovations (F/G/H running now)
6. **Progressive Layer Freezing** — more decoder steps during warmdown
7. **Hyper-Connections** — learned cross-layer mixing
8. **Logit Ensemble** — average logits from multiple checkpoints at eval

### Tier 4: Moonshots (high risk, high reward)
9. **BitNet b1.58 + depth recurrence** — 4x effective params in 16MB, if the x0 anchor works
10. **Arithmetic coding + 8 KV heads** — if coding saves enough bytes to fit MHA

### Dropped
- ~~KV heads=8~~ — doesn't fit in 16MB under current compression
- ~~MLP hidden=1792~~ — doesn't fit in 16MB
- ~~Sigmoid skip gates~~ — hurt in our architecture (run6 regression)
- ~~Decoder 2x LR~~ — hurt in our architecture (run6 regression)
- ~~Neural cache eval~~ — fundamentally broken in forward_logits_cached
