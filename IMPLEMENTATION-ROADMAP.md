# Parameter Golf Implementation Roadmap
## Chiron — Comprehensive Planning Agent
### Date: 2026-03-24 | Deadline: April 30, 2026

---

## Executive Summary

**Goal:** Beat current #1 score of 1.1228 bpb on the OpenAI Parameter Golf leaderboard.
**Target:** Sub-1.10 bpb (conservative), sub-1.05 bpb (stretch), sub-1.00 bpb (moonshot).
**Budget:** ~$20 RunPod = ~7 runs on 8×H100 (or ~28 on 1×H100).
**Timeline:** 5 weeks (March 24 – April 30).

The competition rewards stacking many small improvements. The #1 entry (signalrush, 1.1228 bpb) achieved its score not through one breakthrough but through 8+ techniques stacked together. Our plan follows the same philosophy: implement changes incrementally, validate cheaply on 1×H100, then commit expensive 8×H100 runs only when confident.

---

## Current State Analysis

### Leaderboard Snapshot (as of 2026-03-24)

| Rank | Score (bpb) | Key Techniques |
|------|------------|----------------|
| **#1** | **1.1228** | 11L, EMA, GPTQ-lite, warmdown3500, QAT@0.15 |
| #2 | 1.1248 | 11L, Partial RoPE (16/64), LN scale, EMA, XSA4 |
| #3 | 1.1271 | 11L, XSA4, EMA, Int6 MLP3x, WD=0.04 |
| #4 | 1.1307 | 11L, Efficient Partial XSA (3 layers) |
| #5 | 1.1428 | 10L, Int5/Int6 mixed, BigramHash(10240) |
| #6 | 1.1458 | MLP3x, SmearGate, BigramHash, OrthoInit |
| Baseline | 1.2244 | 9L, 512d, GQA, TiedEmb |

### What's in the Baseline Code (train_gpt.py)

The baseline already includes:
- **Architecture:** 9-layer transformer, 512-dim, 8 heads / 4 KV heads (GQA), 2x MLP, tied embeddings
- **U-Net skips:** First half stores skip connections, second half adds them back (with learnable weights)
- **Optimizer:** Muon (Newton-Schulz orthogonalization) for matrix params, Adam for scalars/embeddings
- **Quantization:** int8 per-row quantization + zlib compression
- **Training:** 524K tokens/step, 20K iterations, ~10 min cap on 8×H100

### What #1 Added on Top of Baseline

The signalrush entry stacked:
1. **11 layers** (instead of 9) — more depth, rebalanced byte budget
2. **EMA** (exponential moving average) — smoother weights, better compression
3. **GPTQ-lite** — post-training quantization clip search (tests multiple clipping percentiles, keeps best)
4. **warmdown3500** — extended warmdown schedule (last 3500 steps reduce LR)
5. **QAT@0.15** — quantization-aware training at 0.15 fraction of training
6. **3x MLP** — expanded feed-forward (hidden = 3 × model_dim)
7. **SmearGate** — gate mechanism that "smears" information across attention heads
8. **BigramHash** — encodes bigram statistics as additional input features
9. **int6 weights** — uses 6-bit quantization instead of int8 for weight storage
10. **zstd-22** — maximum compression ratio (level 22) instead of zlib

---

## Phase 0: Foundation (Week 1 — March 24-30)
### *Goal: Get a working local dev loop and establish baselines*

**No 8×H100 runs yet. All work on 1×H100 or local MLX.**

#### Phase 0A: Environment Setup (Day 1-2)

**Tasks:**
1. Fork the `openai/parameter-golf` repo to our GitHub account
2. Clone locally and on RunPod
3. Download FineWeb dataset: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10` (smoke test subset)
4. Run baseline on 1×H100 to verify setup:
   ```bash
   RUN_ID=baseline_verify \
   ITERATIONS=500 \
   VAL_LOSS_EVERY=100 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```
5. Confirm we reproduce ~1.22 bpb on the baseline

**Deliverable:** Confirmed baseline reproduction on our hardware.

#### Phase 0B: Baseline Profiling (Day 2-3)

**Tasks:**
1. Profile the baseline to understand where bytes go:
   - Parameter count by component (embeddings, attention, MLP, norms)
   - Compression ratio by component (int8 + zlib)
   - Which components dominate the 16MB budget
2. Profile training throughput:
   - Tokens/second per GPU
   - How much of the 10-minute budget is actual training vs. validation
   - Where wall-clock time is spent
3. Run the baseline with `VAL_LOSS_EVERY=200` on full dataset (80 shards) on 1×H100, record final bpb

**Deliverable:** Quantitative profile document of baseline's byte and time budget.

#### Phase 0C: Code Architecture Fork (Day 3-5)

**Tasks:**
1. Create a feature branch: `experiment/stacked-improvements`
2. Build a clean, modular version of `train_gpt.py` that supports:
   - Environment variable toggles for each technique (e.g., `USE_EMA=1`, `USE_BIGRAMHASH=1`)
   - Clean separation of quantization pipeline
   - Logging that records per-component byte usage
3. Add validation harness: script that runs 3 short training runs and reports mean/std bpb (for statistical significance)

**Deliverable:** Modular codebase with toggle-able features.

---

## Phase 1: Low-Hanging Fruit (Week 1-2 — March 27 - April 6)
### *Goal: Implement the techniques proven by #1-#6 on the leaderboard*
### *Expected gain: 1.1228 → 1.08-1.10 bpb*
### *Compute cost: 1×H100 for testing, 0 × 8×H100*

These are proven techniques with known implementations. Implement them first — highest ROI, lowest risk.

### Technique Stack Order (by expected bpb gain per implementation effort):

#### 1.1 EMA (Exponential Moving Average) — ~0.005-0.01 bpb gain
**Why first:** Zero risk, easy to implement, proven by every top entry.
**Code changes:**
```python
# Add EMA state to model weights
ema_decay = 0.999  # or sweep 0.995-0.9999
ema_state = {n: p.data.clone() for n, p in model.named_parameters()}
# During training, after each optimizer step:
for n, p in model.named_parameters():
    ema_state[n].mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
# At eval/export time, swap in EMA weights
```
**Hyperparams to sweep:** `ema_decay` in [0.995, 0.997, 0.999, 0.9995, 0.9999]
**Fallback:** If EMA hurts (unlikely), just use final checkpoint. No regression risk.

#### 1.2 11 Layers + 3x MLP — ~0.01-0.02 bpb gain
**Why second:** Proven by #1-#3. Simple parameter rebalancing.
**Code changes:**
- `NUM_LAYERS=11` (was 9) — adds 2 transformer blocks
- `MLP_MULT=3` (was 2) — triples hidden dim in feed-forward
- Must rebalance: going from 9→11 layers costs ~22% more bytes in transformer blocks. Compensate by reducing other components if needed.
- The baseline's U-Net skip architecture already handles variable layer counts. Verify skip pairing works correctly with 11 layers (5 encoder, 6 decoder or vice versa).

**Hyperparams:** `NUM_LAYERS=11`, `MLP_MULT=3`
**Fallback:** If 11 layers exceeds 16MB after compression, stay at 10 layers.

#### 1.3 Warmdown Schedule — ~0.005-0.01 bpb gain
**Why third:** Proven by #1, simple learning rate schedule change.
**Code changes:**
```python
# Extend warmdown from 1200 to 3500 iterations
WARMDOWN_ITERS=3500
# In the learning rate schedule, add cosine warmdown:
# Phase 1: warmup (20 steps)
# Phase 2: constant LR (iterations - warmdown_iters)
# Phase 3: cosine decay to 0 (last warmdown_iters steps)
```
**Hyperparams to sweep:** `WARMDOWN_ITERS` in [2000, 3500, 5000]

#### 1.4 Int6 Quantization — ~0.01-0.02 bpb gain (frees bytes for more params)
**Why fourth:** Frees ~25% of weight bytes vs int8, allowing larger model in same 16MB.
**Code changes:**
- Modify `quantize_state_dict_int8` → add `quantize_state_dict_int6` path
- int6 means 64 values per weight (-32 to +31), stored 10 weights per 64 bits (60 bits) with 4 bits padding
- Or use nibble packing: 2 nibbles (4 bits each) per weight value, 16 values in 64 bits
- Actually, for simplicity: pack 10 int6 values into 60 bits (= 7.5 bytes), 64 weights into 480 bits (= 60 bytes)
- Per-row scales stored in fp16
- Must handle GPTQ-lite clip search at 6-bit precision

```python
def quantize_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    # Similar to int8 but with 6-bit range: -32 to +31
    t32 = t.float()
    clip_abs = torch.quantile(t32.abs(), 0.999, dim=1)  # ~99.9th percentile
    clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
    scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
    q = torch.clamp(torch.round(clipped / scale[:, None]), -32, 31).to(torch.int8)
    return q, scale.to(dtype=torch.float16)
```

**Fallback:** If int6 + GPTQ-lite doesn't fit cleanly, use mixed int6/int8 (int6 for large weight matrices, int8 for small control tensors).

#### 1.5 GPTQ-lite Post-Training Quantization — ~0.005 bpb gain
**Why fifth:** Refines the quantization to minimize reconstruction error.
**Code changes:**
```python
def gptq_lite_search(weight_tensor, num_trials=20):
    """Try different clipping percentiles, keep the one with lowest reconstruction error."""
    best_clip = 0.999
    best_error = float('inf')
    for p in np.linspace(0.99, 0.999999, num_trials):
        clip_abs = torch.quantile(weight_tensor.abs(), p, dim=1)
        clipped = torch.clamp(weight_tensor, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1e-8)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
        reconstruction = q * scale[:, None]
        error = ((weight_tensor - reconstruction) ** 2).sum().item()
        if error < best_error:
            best_error = error
            best_clip = p
    return best_clip
```

#### 1.6 Sliding Window Evaluation — ~0.01 bpb gain
**Why sixth:** Free improvement — just changes how eval is computed.
**Code changes:**
- Instead of one-shot forward pass through validation, use sliding window with stride < seq_len
- Overlapping contexts give better loss estimates
- `stride = 64` or `128` (vs. full seq_len = 1024)
- Must handle KV-cache or just recompute (recompute is fine for eval)

#### 1.7 QAT (Quantization-Aware Training) — ~0.005-0.01 bpb gain
**Why seventh:** Trains the model knowing it will be quantized, reducing quantization damage.
**Code changes:**
```python
# In the forward pass during QAT phase:
if step > total_iters * (1 - qat_fraction):
    # Simulate quantization noise during forward pass
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point and param.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            q, scale = fake_quantize(param, bits=6)
            param.data = (q.float() * scale).to(param.dtype)
```
**Hyperparams:** `QAT_FRACTION=0.15` (last 15% of training)

---

### Phase 1 Validation Plan

**Before any 8×H100 run:**
1. Implement each technique individually on 1×H100
2. Run 500-iteration smoke tests to verify no regressions
3. Run full 10-minute 1×H100 training to estimate bpb impact
4. Only stack techniques that individually showed positive or neutral impact

**Testing cadence:**
```
Day 1-2: Implement EMA → test (1×H100, 500 iters)
Day 2-3: Implement 11L + 3xMLP → test
Day 3-4: Implement warmdown → test
Day 4-5: Implement int6 → test
Day 5-6: Implement GPTQ-lite → test
Day 6-7: Stack all together → test (1×H100, full run)
Day 7: First 8×H100 validation run (Run #1)
```

**Run #1 (8×H100):** Validate stacked Phase 1 improvements. Target: ≤1.10 bpb.
- Cost: ~$2.86 (1/7 of budget)
- If score ≥ 1.10: proceed to Phase 2
- If score > 1.12: debug and fix before proceeding

---

## Phase 2: Advanced Techniques (Week 2-3 — April 6-13)
### *Goal: Add techniques used by top entries that require more careful implementation*
### *Expected gain: 1.10 → 1.06-1.08 bpb*
### *Compute cost: 2 × 8×H100 runs*

#### 2.1 Partial RoPE (Rotary Position Embedding) — ~0.005-0.01 bpb gain
**What:** Apply RoPE to only a subset of dimensions in each attention head.
**Code changes:**
```python
class Rotary(nn.Module):
    def __init__(self, dim: int, rope_dims: int = 16, base: float = 10000.0):
        super().__init__()
        # Only apply RoPE to first `rope_dims` dimensions of the head
        self.rope_dims = rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # ... rest unchanged ...

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Only rotate the first `rope_dims` dimensions
    half = x.size(-1) // 2
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rope_rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rope_rotated, x_pass), dim=-1)
```
**Hyperparams:** `ROPE_DIMS` in [16, 32, 48, 64]. #2 used 16/64 (16 dims out of 64 head_dim).
**Fallback:** Full RoPE (all dims) — just set `ROPE_DIMS = head_dim`.

#### 2.2 Partial XSA (Cross-Stage Attention) on Last N Layers — ~0.01 bpb gain
**What:** In the decoder half, let the last few layers attend across the full sequence (not just causal). This is the "cross-attention" in the U-Net structure.
**Code changes:**
```python
# In Block.forward(), modify attention mask:
if self.use_xsa and is_decoder_layer:
    # Non-causal attention: attend to all positions
    y = F.scaled_dot_product_attention(q, k, v, is_causal=False, ...)
else:
    # Standard causal attention
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, ...)
```
**Hyperparams:** `XSA_LAYERS=4` (apply to last 4 layers). The #1/#2 entries use XSA on last 4.
**Fallback:** All-causal (standard) — no change needed.

#### 2.3 BigramHash — ~0.003-0.005 bpb gain
**What:** Augment token embeddings with bigram frequency statistics.
**Code changes:**
```python
class BigramHash(nn.Module):
    def __init__(self, vocab_size: int, hash_table_size: int = 10240):
        super().__init__()
        self.hash_table_size = hash_table_size
        # Small learned table: hash(token_id, prev_token_id) -> embedding
        self.bigram_embed = nn.Embedding(hash_table_size, model_dim)
        nn.init.zeros_(self.bigram_embed.weight)

    def forward(self, x: Tensor, prev_tokens: Tensor) -> Tensor:
        # hash = (prev_token * vocab_size + token) % hash_table_size
        bigram_ids = (prev_tokens.long() * vocab_size + x.long()) % self.hash_table_size
        return self.bigram_embed(bigram_ids)

# In GPT.forward():
bigram_emb = self.bigram_hash(input_ids, F.pad(input_ids[:, :-1], (1, 0), value=0))
x = x + bigram_emb
```
**Hyperparams:** `BIGRAM_HASH_SIZE` in [4096, 10240, 20480]. #5 used 10240.
**Fallback:** No bigram feature — set hash size to 0 or skip.

#### 2.4 SmearGate — ~0.003-0.005 bpb gain
**What:** A gating mechanism that mixes attention outputs across heads in a "smeared" fashion.
**Code changes:**
```python
# Add a learnable mixing matrix after attention projection
class SmearGate(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        # Small mixing matrix: (num_heads, num_heads) initialized near identity
        self.mix = nn.Parameter(torch.eye(num_heads) * 0.1 + torch.randn(num_heads, num_heads) * 0.01)

    def forward(self, y: Tensor) -> Tensor:
        # y shape: (batch, heads, seq, head_dim)
        # Mix across heads
        bsz, heads, seq, hd = y.shape
        y = y.permute(0, 2, 3, 1).reshape(bsz, seq, heads * hd)
        # Apply head mixing via projection
        y = y.reshape(bsz, seq, heads, hd).permute(0, 2, 1, 3)
        return y
```

#### 2.5 Spectral/Orthogonal Initialization — ~0.003 bpb gain
**What:** Initialize weight matrices with orthogonal or spectral normalization for better training dynamics.
**Code changes:**
```python
def orthogonal_init(tensor, gain=1.0):
    nn.init.orthogonal_(tensor, gain=gain)

# In _init_weights:
for module in self.modules():
    if isinstance(module, CastedLinear):
        orthogonal_init(module.weight.data, gain=math.sqrt(2))
```

---

### Phase 2 Validation Plan

**Run #2 (8×H100):** Phase 1 stack + Partial RoPE. Target: confirm ~0.005 bpb gain.
**Run #3 (8×H100):** Phase 1 stack + Partial RoPE + XSA4 + BigramHash. Target: ≤1.08 bpb.

**Before each 8×H100 run:**
- 1×H100 smoke test (500 iters) to catch obvious bugs
- 1×H100 full run (10 min) to estimate bpb improvement
- Only commit 8×H100 if 1×H100 run shows positive delta

**Fallback strategy:** If XSA or BigramHash hurt bpb on 1×H100, drop them and proceed with just RoPE + SmearGate.

---

## Phase 3: Novel Techniques (Week 3-4 — April 13-20)
### *Goal: Push beyond what the leaderboard has shown*
### *Expected gain: 1.06 → 1.00-1.03 bpb*
### *Compute cost: 2 × 8×H100 runs*

This is where we try to differentiate. These techniques are NOT proven in the competition but have strong theoretical backing.

#### 3.1 Depth Recurrence (Shared Weights) — ~0.01-0.03 bpb gain
**What:** Instead of 11 unique layers, use 3-4 unique blocks repeated 3 times each. Same effective depth (9-12 layers of computation) with only 3-4 layers of parameters.
**Code changes:**
```python
class GPTWithRecurrence(nn.Module):
    def __init__(self, ..., unique_blocks=4, recurrence_depth=3):
        self.blocks = nn.ModuleList([Block(...) for _ in range(unique_blocks)])
        self.recurrence_depth = recurrence_depth

    def forward(self, x, x0):
        skips = []
        for cycle in range(self.recurrence_depth):
            for i, block in enumerate(self.blocks):
                x = block(x, x0)
                if cycle == 0:
                    skips.append(x)
                else:
                    # Add skip from first pass
                    x = x + self.skip_weights[len(skips) - 1] * skips[len(skips) - 1]
        return x
```

**Tradeoff:** With 4 unique blocks × 3 recurrences = 12 effective layers, but only 4 layers of parameters. This frees ~60% of the transformer block parameters, which can be reallocated to:
- Larger model_dim (576 or 608 instead of 512)
- Larger MLP (3x or 4x)
- More attention heads

**Hyperparams:** `UNIQUE_BLOCKS=4`, `RECURRENCE_DEPTH=3`, reallocate freed bytes to `MODEL_DIM=576`.

**Fallback:** If recurrence destabilizes training, fall back to 11 unique layers (Phase 2 configuration).

#### 3.2 Low-Rank Training (LoRA-Style Parameterization) — ~0.005-0.015 bpb gain
**What:** Train large weight matrices as products of two smaller matrices: W ≈ A × B, where A is (d, r) and B is (r, d) with r << d.
**Code changes:**
```python
class LowRankLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, out_dim, bias=False)
        nn.init.orthogonal_(self.a.weight)
        nn.init.orthogonal_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(x))
```

**Application:** Apply to MLP layers where the 3x hidden dim creates the largest weight matrices.
- Standard 512→1536 MLP: 512×1536 = 786K params per layer
- Low-rank: 512×256 + 256×1536 = 131K + 393K = 524K params (33% savings)
- Freed bytes go to more layers or larger dim

**Fallback:** Full-rank matrices (standard linear layers).

#### 3.3 BitNet-Style 1.58-bit Ternary Weights for MLP — ~0.02-0.05 bpb gain
**What:** Replace MLP weight storage with ternary (-1, 0, +1) values. This is the most aggressive technique we'll try.
**Code changes:**
```python
class BitLinear(nn.Module):
    """Ternary-weight linear layer with learned scaling factor."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        # During forward: compute ternary weights via absmean quantization
        w_mean = self.weight.abs().mean()
        w_scaled = self.weight / (w_mean + 1e-8)
        w_ternary = torch.round(torch.clamp(w_scaled, -1, 1))
        # Forward with ternary weights
        return F.linear(x, w_ternary * w_mean.to(x.dtype), self.bias)

    def get_quantized_weight(self):
        w_mean = self.weight.abs().mean()
        w_ternary = torch.round(torch.clamp(self.weight / (w_mean + 1e-8), -1, 1))
        return w_ternary, w_mean
```

**Storage:** Each ternary weight needs 2 bits. A 512×1536 MLP matrix = 786K params × 2 bits = 196KB (vs. 786KB for int8, vs. 589KB for int6). ~3-4x compression over int6!

**Fallback:** If ternary MLP destabilizes training, apply only to the output projection of MLP (smaller impact but still saves bytes).

#### 3.4 Enhanced Compression Pipeline — ~0.005-0.01 bpb gain
**What:** Use zstd-22 with trained dictionary, or try BWT + zstd, or zstd-22 with block-level tuning.
**Code changes:**
```python
import zstandard as zstd

def compress_artifact(model_bytes: bytes) -> bytes:
    # Option A: zstd-22 with a trained dictionary
    dict_data = zstd.ZstdCompressionDict(calibration_data, level=22)
    cctx = zstd.ZstdCompressor(level=22, dict_data=dict_data)
    return cctx.compress(model_bytes)

    # Option B: Split model into chunks and compress each with zstd-22
    # Control tensors (scales, norms) compress better separately from weight matrices
```

**Fallback:** Standard zstd-22 without dictionary (already good).

---

### Phase 3 Validation Plan

**Run #4 (8×H100):** Phase 2 stack + Depth Recurrence. Target: ≤1.06 bpb.
**Run #5 (8×H100):** Phase 2 stack + Depth Recurrence + Enhanced Compression. Target: ≤1.05 bpb.

**Critical decision point:** After Run #4, evaluate whether depth recurrence works. If it doesn't:
- Shift budget to more runs of Phase 2 stack + BitNet MLP
- Try low-rank training instead

---

## Phase 4: Final Tuning & Submission (Week 4-5 — April 20-30)
### *Goal: Squeeze every last 0.001 bpb and prepare submission*
### *Expected gain: 1.03 → <1.00 bpb (stretch)*
### *Compute cost: 2 × 8×H100 runs*

#### 4.1 Hyperparameter Fine-Tuning
**Sweep the most sensitive hyperparameters on 1×H100:**

| Hyperparameter | Range | Sensitivity |
|----------------|-------|-------------|
| `MODEL_DIM` | [480, 512, 544, 576, 608] | High — affects all components |
| `MLP_MULT` | [2.5, 3.0, 3.5, 4.0] | Medium |
| `NUM_LAYERS` | [10, 11, 12, 13] | High |
| `ROPE_DIMS` | [8, 16, 32, 48] | Medium |
| `XSA_LAYERS` | [2, 3, 4, 5] | Medium |
| `EMA_DECAY` | [0.995, 0.999, 0.9995, 0.9999] | Low |
| `WARMDOWN_ITERS` | [2000, 3500, 5000, 7000] | Medium |
| `QAT_FRACTION` | [0.1, 0.15, 0.2, 0.25] | Medium |
| `EMBED_LR` | [0.3, 0.5, 0.6, 0.8] | Medium |
| `MATRIX_LR` | [0.02, 0.04, 0.06, 0.08] | Medium |
| `LR` (global) | [0.01, 0.04, 0.1] | High |
| `BIGRAM_HASH_SIZE` | [0, 4096, 10240, 20480] | Low |

#### 4.2 Quantization Bit Width Sweep
Try mixed precision strategies:
- All int6 (proven)
- MLP int4 / attention int6 / embed fp16
- MLP BitNet ternary / attention int6 / embed int8
- Block-wise variable precision (more important layers get more bits)

#### 4.3 Statistical Validation
**Requirement:** p < 0.01 with ≥ 0.005 nats improvement.
```
# Run 5+ training runs with different seeds
for seed in [1337, 42, 7, 12345, 99999]:
    RUN_ID=final_seed_${seed} SEED=${seed} <full command>
# Compute mean and std of final bpb across runs
# Verify p-value < 0.01 vs. current #1
```

#### 4.4 Submission Package
1. Ensure `train_gpt.py` is self-contained and under 1500 lines
2. Run final submission verification:
   ```bash
   # Full 8×H100 run with logging
   RUN_ID=final_submission \
   torchrun --nproc_per_node=8 train_gpt.py
   # Check artifact size < 16,000,000 bytes
   # Verify val_bpb and val_loss in logs
   ```
3. Create PR with:
   - `train_gpt.py` (clean, documented)
   - Training logs (multiple runs for statistical significance)
   - `submission.json`
   - `README.md` with technique write-up
4. Submit PR before April 30

---

### Phase 4 Validation Plan

**Run #6 (8×H100):** Final tuned configuration, seed 1337.
**Run #7 (8×H100):** Final tuned configuration, seed 42 (for statistical validation).

**Additional 1×H100 runs:** As many as needed for hyperparameter sweeps. Budget ~$10-15 for additional 1×H100 runs if needed.

---

## Compute Budget Allocation

| Run | GPU Config | Phase | Purpose | Est. Cost |
|-----|-----------|-------|---------|-----------|
| — | 1×H100 | 0-1 | Smoke tests, debugging | ~$0.40 (20 min) |
| — | 1×H100 | 1-2 | Full training, technique validation | ~$2.00 (×5 runs) |
| **#1** | 8×H100 | 1 | Validate Phase 1 stack | ~$2.86 |
| — | 1×H100 | 2 | Phase 2 smoke tests | ~$0.80 |
| **#2** | 8×H100 | 2 | Phase 1 + Partial RoPE | ~$2.86 |
| **#3** | 8×H100 | 2 | Full Phase 2 stack | ~$2.86 |
| — | 1×H100 | 3 | Phase 3 smoke tests | ~$0.80 |
| **#4** | 8×H100 | 3 | Phase 2 + Depth Recurrence | ~$2.86 |
| **#5** | 8×H100 | 3 | Phase 2 + BitNet (if recurrence fails) | ~$2.86 |
| — | 1×H100 | 4 | Hyperparameter sweeps | ~$3.00 |
| **#6** | 8×H100 | 4 | Final config run | ~$2.86 |
| **#7** | 8×H100 | 4 | Statistical validation (2nd seed) | ~$2.86 |
| | | | **Total** | **~$20.00** |

---

## Fallback Plans

### If Phase 1 only reaches 1.12+ (no improvement over #1):
- **Plan B1:** Increase model_dim to 576 at expense of layers (10L × 576d vs 11L × 512d)
- **Plan B2:** Try int4 quantization for MLP weights only (aggressive but may work)
- **Plan B3:** Add test-time training (LoRA-TTT as proven by #14 entry, 1.1928 bpb)

### If Depth Recurrence (Phase 3) doesn't work:
- **Plan C1:** Go all-in on BitNet ternary weights for MLP layers
- **Plan C2:** Try product quantization (codebook-based) instead
- **Plan C3:** Explore Mamba/SSM hybrid — replace last 2 attention layers with Mamba blocks

### If we can't break 1.05 bpb:
- **Plan D1:** Increase sequence length to 2048 (more context, better predictions)
- **Plan D2:** Add test-time compute (self-distillation, chain-of-thought-style reasoning)
- **Plan D3:** Experiment with RWKV-style linear attention for some layers

### If we blow the 16MB budget:
- **Immediate fix:** Reduce model_dim by 16 (512→496 or 480)
- **Second fix:** Reduce vocab_size (but this changes tokenizer — needs careful bpb recalculation)
- **Third fix:** Use more aggressive quantization (int5 or int4 for MLP)

---

## Testing Strategy

### The 1×H100 Validation Protocol

**Every technique must pass these gates before touching 8×H100:**

1. **Smoke test (500 iterations, ~1 min):**
   - Does training converge? (loss decreases)
   - No NaN or Inf losses?
   - Artifact size is reasonable?

2. **Medium test (2000 iterations, ~4 min):**
   - Is val_bpb better than baseline equivalent?
   - Compare: same config without the new technique
   - Accept if delta ≥ -0.002 (no worse than -0.002 bpb)

3. **Full test (full 10 min on 1×H100):**
   - Compare final val_bpb against previous best 1×H100 run
   - Note: 1×H100 scores are ~0.01-0.03 bpb worse than 8×H100 (less data per step)
   - Use delta (change in bpb) as the signal, not absolute value

### Multi-Seed Validation

Before submitting:
- Run 3+ seeds for the final configuration
- Compute mean bpb and standard deviation
- Verify: mean_bpb + 2*std_bpb < 1.1228 (current #1) with p < 0.01
- Use Welch's t-test or bootstrap resampling

### Artifact Size Monitoring

After every config change, check artifact size:
```bash
# Compress and check size
python -c "
import torch, io, zlib
state = torch.load('checkpoint.pt')
buf = io.BytesIO()
torch.save(state, buf)
compressed = zlib.compress(buf.getvalue(), level=22)
print(f'Compressed size: {len(compressed):,} bytes')
print(f'Headroom: {16_000_000 - len(compressed):,} bytes')
"
```
Keep ≥ 100KB headroom for code (train_gpt.py is ~50KB).

---

## Week-by-Week Timeline

### Week 1 (March 24-30): Foundation + Phase 1
| Day | Task | Owner | GPU |
|-----|------|-------|-----|
| Mon | Fork repo, setup RunPod, download data | — | — |
| Tue | Run baseline, profile byte budget | — | 1×H100 |
| Wed | Implement EMA + 11L + 3xMLP + warmdown | — | 1×H100 |
| Thu | Implement int6 + GPTQ-lite | — | 1×H100 |
| Fri | Stack all Phase 1, full 1×H100 run | — | 1×H100 |
| Sat | Implement sliding eval + QAT | — | 1×H100 |
| Sun | **Run #1: 8×H100 Phase 1 validation** | — | **8×H100** |

### Week 2 (March 31 - April 6): Phase 2
| Day | Task | Owner | GPU |
|-----|------|-------|-----|
| Mon | Implement Partial RoPE | — | 1×H100 |
| Tue | Implement Partial XSA + BigramHash | — | 1×H100 |
| Wed | Implement SmearGate + OrthoInit | — | 1×H100 |
| Thu | Stack Phase 2, smoke test | — | 1×H100 |
| Fri | **Run #2: Phase 1 + RoPE (8×H100)** | — | **8×H100** |
| Sat | Debug if needed | — | 1×H100 |
| Sun | **Run #3: Full Phase 2 (8×H100)** | — | **8×H100** |

### Week 3 (April 7-13): Phase 3
| Day | Task | Owner | GPU |
|-----|------|-------|-----|
| Mon | Implement depth recurrence | — | 1×H100 |
| Tue | Implement BitNet ternary MLP | — | 1×H100 |
| Wed | Implement low-rank training | — | 1×H100 |
| Thu | Smoke test Phase 3 techniques | — | 1×H100 |
| Fri | **Run #4: Phase 2 + recurrence (8×H100)** | — | **8×H100** |
| Sat | Decision: recurrence or BitNet path? | — | — |
| Sun | **Run #5: Chosen Phase 3 path (8×H100)** | — | **8×H100** |

### Week 4 (April 14-20): Tuning
| Day | Task | Owner | GPU |
|-----|------|-------|-----|
| Mon-Wed | Hyperparameter sweeps on 1×H100 | — | 1×H100 |
| Thu | Quantization bit-width experiments | — | 1×H100 |
| Fri | Select final configuration | — | — |
| Sat-Sun | Multi-seed validation (3+ runs) | — | 1×H100 |

### Week 5 (April 21-30): Final Runs + Submission
| Day | Task | Owner | GPU |
|-----|------|-------|-----|
| Mon | **Run #6: Final config (8×H100)** | — | **8×H100** |
| Tue | **Run #7: Statistical validation (8×H100)** | — | **8×H100** |
| Wed-Thu | Prepare PR submission | — | — |
| Fri | Review and polish write-up | — | — |
| Sat-Mon | Buffer for debugging/re-runs | — | 1×H100 |
| Tue Apr 28 | **Submit PR** | — | — |
| Wed-Thu | Buffer for PR review/comments | — | — |
| Fri Apr 30 | **Deadline** | — | — |

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 16MB budget too tight for 11L+ | Medium | High | Reduce dim to 480, or use 10L |
| Depth recurrence destabilizes training | Medium | Medium | Fallback to unique layers + BitNet |
| BitNet ternary too aggressive | High | Medium | Use only for MLP output projection |
| EMA adds too much eval time | Low | Low | Skip EMA if time-constrained |
| Can't reproduce #1's score exactly | Medium | High | Follow their exact config first, then improve |
| Leaderboard moves while we're building | High | Medium | Monitor weekly, adjust target |
| zstd-22 compression too slow for 10-min cap | Low | Low | Use zstd-19 as fallback |
| RunPod availability issues | Low | High | Have backup GPU provider ready |

---

## Quick Reference: Technique Impact Summary

| Technique | Est. bpb Δ | Risk | Implementation Effort | Proven? |
|-----------|-----------|------|----------------------|---------|
| EMA | -0.008 | Very Low | Low | ✅ #1-#6 |
| 11L + 3xMLP | -0.015 | Low | Low | ✅ #1-#6 |
| Warmdown 3500 | -0.007 | Very Low | Low | ✅ #1 |
| Int6 quantization | -0.015 | Low | Medium | ✅ #1-#3 |
| GPTQ-lite | -0.005 | Low | Medium | ✅ #1 |
| Sliding eval | -0.010 | Very Low | Low | ✅ #11-#14 |
| QAT @ 0.15 | -0.007 | Low | Medium | ✅ #1 |
| Partial RoPE (16/64) | -0.008 | Low | Medium | ✅ #2 |
| Partial XSA (last 4) | -0.010 | Low | Medium | ✅ #2-#4 |
| BigramHash | -0.004 | Low | Low | ✅ #5-#6 |
| SmearGate | -0.004 | Low | Medium | ✅ #6 |
| OrthoInit | -0.003 | Low | Low | ✅ #6 |
| **Depth Recurrence** | **-0.020** | **Medium** | **High** | ❌ **Novel** |
| **BitNet Ternary MLP** | **-0.030** | **High** | **High** | ❌ **Novel** |
| **Low-Rank Training** | **-0.010** | **Medium** | **Medium** | ❌ **Novel** |
| **Enhanced Compression** | **-0.008** | **Low** | **Medium** | ❌ **Partially novel** |

**Conservative estimate:** Proven techniques only → ~1.09-1.10 bpb
**Moderate estimate:** + Depth Recurrence → ~1.06-1.08 bpb
**Aggressive estimate:** + BitNet Ternary → ~1.02-1.05 bpb
**Moonshot:** All techniques + perfect tuning → <1.00 bpb

---

## Notes

- Monitor the leaderboard weekly. If new techniques appear, assess whether to pivot.
- The competition rules allow "external compute" for hyperparameter tuning — this is explicitly OK per the FAQ.
- Submissions must beat current SOTA by ≥ 0.005 nats to qualify for the leaderboard.
- Statistical significance: p < 0.01, requiring multiple runs with different seeds.
- The artifact includes train_gpt.py code bytes + compressed model bytes. Code optimization (shorter variable names, fewer comments, code golf) can free ~5-10KB for model weights.

---

*Roadmap created by Chiron 🐴 | Comprehensive Planning Agent*
*Reviewed against: ARCHITECTURE-ANALYSIS.md, RESEARCH-FINDINGS.md, LANDSCAPE-RESEARCH.md*
*Last updated: 2026-03-24*
