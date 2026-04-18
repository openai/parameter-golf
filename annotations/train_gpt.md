# parameter-golf `train_gpt.py` — comparative reading notes

**Date:** 2026-04-15 (afternoon session)
**Source:** `projects/parameter-golf/parameter-golf-upstream/train_gpt.py` (1126 lines, commit current as of clone)
**Session type:** interactive, `learn-from-paper` style — reading `train_gpt.py` with nanoGPT `model.py` as the reference baseline. Every observation is framed as a **diff from nanoGPT** and a **"why" for the 16 MB / 10-min context**.

---

## Quick-reference diff preview (to fill in as we read)

| Axis | nanoGPT `model.py` | parameter-golf `train_gpt.py` | Why it matters for PG |
|---|---|---|---|
| Normalization | `LayerNorm` (mean-center + γ/β) | **`RMSNorm`** — no mean-center, no γ/β. 0 params. | Save ~38K params at C=768; affine recovered cheaper at Block level via `attn_scale`/`mlp_scale` |
| Position encoding | learned `wpe` (~786K params) | **Rotary (RoPE)** — 0 params, no block_size ceiling, relative position in dot product | Save 786K params + length generalization |
| Linear layer | `nn.Linear` (single dtype) | **`CastedLinear`** — store fp32, cast to bf16 at matmul time | bf16 matmul ~2–4× faster on H100; fp32 storage protects optimizer's small updates from rounding |
| Optimizer | AdamW | **Muon** — orthogonalized gradients for 2D matrices | Better matrix optimization, faster convergence |
| State dict | full precision | **INT8 + zlib** | 4× compression → fits 16 MB |
| Tokenizer | GPT-2 BPE / tiktoken | **SentencePiece** pluggable (SP1024/4096/8192) | Vocab size as tunable lever |
| Eval | single-pass CE | **sliding-window bpb** — tokenizer-agnostic | Fair comparison across tokenizers |
| Data | numpy memmap | sharded + distributed `DistributedTokenLoader` | Multi-GPU training |
| Dropout | optional, default 0 | **gone entirely** | minor code savings |
| Biases | optional, default on | **none on any Linear** (always `bias=False`) | bias-redundancy argument from nanoGPT session: next op has its own learnable params; LN-bias / Linear-bias are redundant |

---

## Segment 1 — RMSNorm (L500–508)

**Code:**
```python
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
```

**Diff from nanoGPT's LayerNorm:**
- **No `weight` (γ).** No per-channel learnable scale.
- **No `bias` (β).** No per-channel learnable shift.
- **No mean-centering** in the math.
- → **Zero learnable parameters.** Each instance just calls `F.rms_norm`.
- `eps` is a constructor arg defaulting to `None` (PyTorch then uses its dtype-default; previously hardcoded `1e-5` in nanoGPT). Same role: prevent division-by-zero when input variance is degenerate.

**Math:**
$$
y_i = \frac{x_i}{\sqrt{\frac{1}{C}\sum_j x_j^2 + \varepsilon}}
$$
"Rescale each C-dim vector to unit root-mean-square magnitude." Compare LayerNorm: $y_i = (x_i - \mu)/\sqrt{\sigma^2 + \varepsilon} \cdot \gamma_i + \beta_i$. RMSNorm drops the mean-center and the affine.

**Why it matters for PG:**
- nanoGPT had ~25 LayerNorms (2 per Block + 1 final), each with `2·C ≈ 1536` params at C=768 → ~38K params total just for LN affines.
- RMSNorm spends 0 on these.
- Per-channel scale capability is **recovered cheaper** later: each Block has `attn_scale` + `mlp_scale` (each shape `(dim,)`). 2·C per Block instead of LN's 4·C per Block — half the cost, same job.
- Mean-centering is **fine to drop**: empirically networks train fine without it (LLaMA proved this; everyone does it now). The token's mean carries info that the network can use.

**Notes / open questions:**
- Is `eps=None` the same as PyTorch's default for all dtypes, or does it depend? (Marginal; matters only at extreme low precision.)

---

## Segment 2 — CastedLinear + `restore_low_dim_params_to_fp32` (L509–521)

**Code:**
```python
class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
```

**Diff from nanoGPT's `nn.Linear`:**
- Inherits from `nn.Linear` (gets `__init__`, `weight`, `bias`, etc., for free); only overrides `forward`.
- `forward` casts `self.weight` (and bias if any) to `x.dtype` **at matmul time**.
- Storage stays at whatever dtype `self.weight` was created in (typically fp32). The cast produces a temporary bf16 tensor for the matmul; the stored fp32 is untouched.

**Mixed-precision rationale:**
- Optimizer (Muon) needs fp32 weights so tiny updates (~`1e-5`) actually accumulate into ~`1.0`-magnitude weights without being rounded away.
- Matmul on H100 tensor cores runs ~2–4× faster in bf16 than fp32.
- bf16 vs fp16: bf16 has the **same exponent range** as fp32 (8 exponent bits) but less mantissa precision (~2–3 decimal digits). Critical: bf16 doesn't overflow on the dynamic range LLM gradients/activations span, where fp16 would.

**Sidecar function `restore_low_dim_params_to_fp32`:**
- Selectively promotes 1D params (or named "control" tensors) back to fp32.
- Used after a `model.to(bf16)`-style cast to keep small precious params (norms, scales, gains) in fp32 while leaving the big 2D weights in bf16.

**Why it matters for PG:**
- **Training speed**, not artifact size. CastedLinear affects what runs during the 10-min training window; doesn't shrink the saved checkpoint.
- The 16 MB checkpoint shrink comes from a **separate** mechanism — INT8 quantization at save/load time (Segment 7).
- Convention throughout the file: every `Linear` is a `CastedLinear` with `bias=False`.

**Bonus: `_zero_init` attribute (mentioned but not in the class):**
- Throughout the file you'll see `self.proj._zero_init = True` set on certain Linears.
- In `GPT._init_weights()`, this flag triggers `nn.init.zeros_(module.weight)` instead of the default Kaiming.
- Applied to the **output projections** that write into the residual stream (`proj` in attention, `proj` in MLP, `lm_head`).
- Effect: at step 0, every Block contributes literally zero to the residual stream → the whole network is the identity at init → trains stably no matter how deep. nanoGPT achieved similar with a scaled-random init (`std=0.02/√(2·n_layer)`); parameter-golf is more aggressive (literal zero).

---

## Segment 3 — Rotary position embeddings (L524–553)

**Code:**
```python
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        ...

    def forward(self, seq_len, device, dtype):
        # cache cos/sin tables of shape (1, 1, seq_len, dim/2)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))   # (seq_len, dim/2)
        self._cos_cached = freqs.cos()[None, None, :, :]
        self._sin_cached = freqs.sin()[None, None, :, :]


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
```

**Diff from nanoGPT's learned `wpe`:**

- `wpe` = `nn.Embedding(block_size, n_embd)` — at `block_size=1024, n_embd=768` → **786,432 learned params**, absolute positions, hard ceiling at `block_size`.
- `Rotary` has **zero learnable parameters**. `inv_freq` is a `register_buffer(persistent=False)` — a tensor owned by the module, not an `nn.Parameter`; not optimized; not saved in the state dict. `_cos_cached` and `_sin_cached` are plain attributes (lazy cache of position-dependent cos/sin tables).
- nanoGPT injects position *once* at the input (`x = tok_emb + pos_emb`) into the residual stream. Rotary injects position *inside each attention head*, by rotating Q and K — never touches V or the residual stream directly.
- The 32 frequencies (for D=64) are geometrically spaced by `inv_freq = 1 / base^(2i/D)` with `base = 10000`. Pair 0 rotates ~1 rad/token (fast, full turn every ~6 tokens); pair 31 rotates ~1.25e-4 rad/token (slow, full turn ~50K tokens). Multi-scale position fingerprint, reusing existing Q/K dims.
- Split-halves pairing convention (this code): pair `i = (x[i], x[i + D/2])`, giving contiguous GPU-friendly slices. Math is equivalent to the original RoPE paper's adjacent-pairs pairing `(x[2i], x[2i+1])` up to a fixed permutation.

**Math of one pair's rotation:**

At position `t`, pair `i = (a, b)` with frequency `ω_i` rotates by `θ = t·ω_i`:
$$
(a', b') = (a\cos\theta + b\sin\theta,\ -a\sin\theta + b\cos\theta)
$$
(Matches the code: `torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), ...)`.)

**Why the dot product gets relative position info:**
If Q_t is the rotated Q at position t and K_s is the rotated K at position s, then for each pair
$$
(a_Q\cos t\omega + b_Q\sin t\omega)(a_K\cos s\omega + b_K\sin s\omega) + (\text{second-half term})
$$
collapses to an expression depending only on $t - s$ (via cos/sin addition formulas). The full dot product `Q_t · K_s` becomes a function of relative offset `(t-s)`, **not** the absolute positions. Mathematical identity:
$$
Q_t \cdot K_s \;=\; Q \cdot R_{(s-t)\omega} \cdot K
$$
where $R_\theta$ is the block-diagonal rotation matrix (one 2×2 rotation per pair). Rotations preserve dot products, so the content geometry is undisturbed.

**Why it matters for PG:**

- **Param savings: ~786K.** Every one of those goes into deeper/wider layers or more vocab — directly cash in the 16 MB budget.
- **No `block_size` ceiling at training.** Training seq length and eval seq length can differ. (Eval uses sliding-window bpb — Segment 8 — so this unlocks longer context at eval even if training stays short.)
- **Content × position coupling is learnable.** A head can learn a Q direction that lines up with rotated K at a specific relative offset → attention heads can specialize on "attend to 3 tokens back," "attend to same-sentence," etc. This is richer than a fixed additive position offset (what you'd get by appending sin/cos coords).
- **Norm preservation.** Rotations keep `||Q||` constant, so the `1/√D` softmax scaling stays calibrated; no need to retune.

**Contrast with "just append sin/cos coords":** appending 2 extra dims `(sin tω, cos tω)` to Q/K *does* give relative position info in the dot product (as `cos((t-s)ω)`) — but (i) only at one frequency unless you grow D a lot, (ii) position contribution is additive and head-independent (can't be modulated by learned content), (iii) bumps ||Q|| and breaks the attention scaling. Rotary wins on all three.

**Notes / open questions:**
- Caching: `Rotary.forward` rebuilds the cos/sin tables whenever `seq_len` or device changes. In practice seq_len is fixed per training run, so it's a one-time cost.
- `persistent=False` on `inv_freq` means it's NOT in `state_dict()` — no artifact-size cost.

---

## Segment 4 — CausalSelfAttention + MLP + Block (L555–647)

### CausalSelfAttention

**Key diffs from nanoGPT:**

**1 — GQA (Grouped Query Attention)**
- nanoGPT: single `c_attn: Linear(C, 3C)` → Q, K, V all same size, all `n_head` heads
- Here: `c_q: CastedLinear(dim, dim)` but `c_k, c_v: CastedLinear(dim, kv_dim)` where `kv_dim = num_kv_heads * head_dim`
- With `num_heads=8, num_kv_heads=4`: K and V matrices are **2× smaller**. Multiple Q heads share the same K/V head.
- Saves params in `c_k`, `c_v` + shrinks KV cache at inference. `enable_gqa=True` in PyTorch handles the broadcasting.

**2 — QK-norm**
```python
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))
```
Applied per-head after reshape, before rotary. Clamps Q and K to unit RMS magnitude → prevents dot products from blowing up → training stability. Zero params (no affine). Not in nanoGPT at all.

**3 — q_gain**
```python
self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
q = q * self.q_gain[None, :, None, None]
```
QK-norm fixes Q's magnitude to ~1, losing control over attention sharpness. `q_gain` (one learned scalar per head) gives it back — heads can independently learn to be sharp (large gain) or diffuse (small gain). Cost: just `num_heads` params (e.g., 8 scalars). Baseline `qk_gain_init=1.5`; top submissions push to 5.25.

**4 — Fused FlashAttention**
```python
y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=...)
```
Replaces nanoGPT's manual 5-line `softmax(QK/√d)V`. Uses FlashAttention under the hood — tiles the computation in SRAM, never materializes the full `(T×T)` attention matrix. 2-4× faster, much less memory. `is_causal=True` bakes the causal mask into the kernel tiling — no mask buffer needed.

**5 — Rotary** (covered in Segment 3) — applied to Q and K after QK-norm.

---

### MLP

```python
def forward(self, x):
    x = torch.relu(self.fc(x))
    return self.proj(x.square())
```

- nanoGPT used GELU. Here: **ReLU²** (`relu(x).square()`).
- Squaring suppresses small positives (0.1→0.01) and amplifies large ones (2.0→4.0) → **sparse activations**. Most neurons near-zero, a few strongly active. Cleaner representations, better generalization.
- Faster than GELU (no `erf` call).
- `mlp_mult=2` (vs nanoGPT's 4) — MLP hidden is only 2× model_dim. Half as wide, big param saving.
- Both `fc` and `proj` are `CastedLinear(bias=False)`. `proj._zero_init=True` → at init, MLP contributes zero to residual stream.

**MLP param cost:** `2 × M × D²` per block. At M=2, D=512: 1.05M params per block — the **biggest single cost** per block (bigger than attention).

---

### Block

```python
def forward(self, x: Tensor, x0: Tensor) -> Tensor:
    mix = self.resid_mix.to(dtype=x.dtype)
    x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = self.attn(self.attn_norm(x))
    x = x + self.attn_scale[None, None, :] * attn_out
    x = x + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x))
    return x
```

**Two diffs from nanoGPT's Block:**

**1 — x0 (original embeddings passed to every block)**
Every Block receives `x0` = the original token embeddings (computed once in GPT.forward, never modified). `resid_mix = [[1,...], [0,...]]` at init → x0 contributes nothing at start, network learns how much to blend in. Direct gradient highway to original token signal at any depth.

**2 — attn_scale / mlp_scale**
Per-channel learned scalars applied to attention and MLP outputs before adding to residual. Since RMSNorm has no affine (Segment 1), these scalars provide the per-channel scale that nanoGPT's LayerNorm γ would have provided — but placed at the residual addition point rather than inside the norm. Cost: 2×D params per block.

**U-Net structure** (in GPT.forward, not Block.forward):
- First `num_encoder_layers = L//2` blocks: run sequentially, each output saved to `skips` list
- Last `num_decoder_layers = L - L//2` blocks: pop from `skips` in reverse order, add weighted skip before each block
- Mirror pairing: encoder layer 0 ↔ decoder last layer, encoder layer k ↔ decoder layer (n_enc - 1 - k)
- `skip_weights`: learned per-layer scalar (shape `(num_skip_weights, D)`), initialized to 1
- Benefit: gradient shortcuts, early encoder features available to late decoder layers

---

## Segment 5 — GPT class (L648–730)

**New vs nanoGPT:**

**1 — U-Net loop** (see Block above)

**2 — Weight tying** (same as nanoGPT)
```python
if self.tie_embeddings:
    logits_proj = F.linear(x, self.tok_emb.weight)
```
Output projection reuses `tok_emb.weight`. Saves `V × D` params (e.g., 1024×512 = 524K). Every modern LLM does this.

**3 — Logit softcap**
```python
logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
```
Caps logits at ±`logit_softcap` (default 30.0). Prevents softmax saturation (one-hot → gradient dies). When logits saturate, `∂L/∂z_correct = p_correct - 1 ≈ 0` — no learning signal. Softcap prevents this. Introduced in Gemini paper. Applied only at the final output, not inside blocks.

**4 — Input RMSNorm**
```python
x = self.tok_emb(input_ids)
x = F.rms_norm(x, (x.size(-1),))   # ← new
x0 = x
```
Normalizes embeddings at input before passing to blocks. Not in nanoGPT.

---

## Segment 6 — Muon optimizer (L96–177)

**What it is:**
A custom optimizer that replaces AdamW for 2D weight matrices. Core step:
1. Compute gradient (standard backprop)
2. Apply momentum (like SGD)
3. **Orthogonalize** the gradient matrix via Newton-Schulz iteration
4. Apply orthogonalized gradient as weight update

**Orthogonalization:** any matrix G = U·S·Vᵀ (SVD). Orthogonalize = replace S with identity → `ortho(G) = U·Vᵀ`. All singular values set to 1. Preserves directions, normalizes magnitudes.

**Why better than AdamW for matrices:**
- AdamW treats every weight as independent scalar — per-element adaptive scaling. Misses matrix geometry.
- Muon treats the weight as a linear transformation. Takes unit-sized steps in *transformation space* — every direction in the gradient gets equal update regardless of magnitude.
- Result: faster convergence, especially early in training. Critical for 10-min budget.

**Newton-Schulz iteration (`zeropower_via_newtonschulz5`):**
```python
a, b, c = (3.4445, -4.7750, 2.0315)
X = G.bfloat16() / (X.norm() + eps)
for _ in range(steps):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```
- "Zeroth power" = orthogonal factor (singular values → 1)
- Each iteration applies a degree-5 polynomial; 5 iterations = degree-5⁵ = degree-3125 approximation of SVD
- Constants `(3.4445, -4.7750, 2.0315)` are the Chebyshev-optimal coefficients for approximating the matrix sign function on [-1,1]
- Avoids expensive full SVD; runs in bf16 for speed

**Separate optimizers:** Muon for 2D matrices (attention/MLP weights). AdamW for 1D params (embeddings, scalars, gains). Different LRs: `matrix_lr=0.04`, `embed_lr=0.6`, `scalar_lr=0.04`.

---

## Segment 7 — Quantization (L310–426)

**What it does:**
At save time: fp32 weights → INT8 (1 byte/param) → zlib compress → 16 MB artifact.
At load time: decompress → dequantize to bf16 → run normally.
Training always uses fp32/bf16 — quantization only at artifact boundary.

**INT8 quantization:**
```
scale = max(|weights|) / 127
quantized = round(weight / scale)    # integer in [-128, 127]
recovered = quantized * scale        # ~original, with rounding error
```
4× compression vs fp32. Precision loss: ~0.1-0.5% in practice (errors average out across large dot products).

**Outlier problem:** one large weight forces a coarse scale for the whole tensor, wasting precision on small weights. Solutions: per-row/per-channel scales, clipping at 99.99% percentile (this code uses `INT8_CLIP_PERCENTILE = 99.99984`).

**Control tensors kept in fp16:** `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights` — small sensitive params kept at higher precision. Defined in `CONTROL_TENSOR_NAME_PATTERNS`.

**Why INT8 specifically:** 1 byte = 256 levels. Enough for weight matrices where errors average out. Top submissions push to INT6 (64 levels, more compression) or INT4 (16 levels, 8× compression vs fp32, needs QAT).

---

## Segment 8 — Tokenizer + eval_val (L180, L219)

**Tokenizer:**
- nanoGPT: GPT-2 BPE via tiktoken, fixed ~50K vocab.
- Here: SentencePiece, pluggable. Default SP1024 (vocab=1024). Top entries use SP4096/SP8192.
- Larger vocab → fewer tokens per byte → lower val_loss (each token covers more text, fewer prediction steps). But bigger embedding table costs params.
- `build_sentencepiece_luts`: precomputes per-token byte counts and space flags for bpb calculation.

**val_bpb metric:**
```python
bits_per_token  = val_loss / log(2)          # nats → bits
tokens_per_byte = total_tokens / total_bytes  # tokenizer compression ratio
val_bpb         = bits_per_token * tokens_per_byte
```
Tokenizer-agnostic: normalizes for how efficiently the tokenizer compresses text. Fair comparison across all vocab sizes. Baseline ~1.2 bpb. Current SOTA ~1.08 bpb.

---

## Segment 9 — Hyperparameters (L39–88)

**Model shape:**
```
vocab_size=1024, num_layers=9, model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=2
```
Total ~17M params. At INT8 → ~17 MB, zlib compresses just under 16 MB.

**Param count formula:**
```
Total = V×D + min(L//2, L-L//2)×D + L × (D²(2 + 2·Hkv/H + 2M) + H + 4D)
```
Dominant term: `L × D² × (2 + 2·Hkv/H + 2M)`. Quadratic in D — **D is the biggest lever**.
Doubling D → 4× more params. Halving D → 4× fewer.

**Training schedule:**
- 20K steps, batch=524K tokens/step → ~10B tokens total (full FineWeb)
- LR: warmup 20 steps → flat → warmdown last 1200 steps (cosine to 0)
- Hard stop at 600s (10 min)

**Key insight:** `train_batch_tokens=524K` is not a 524K-token context window. It's 512 independent sequences of 1024 tokens each, processed in parallel across 8 GPUs.

---

## Segment 10 — Top leaderboard submission

**Submission:** `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`
**Score:** 1.0810 bpb (vs baseline ~1.2) — **-0.12 bpb improvement**

**Key techniques:**
1. **SP8192** — tokenizer upgrade, biggest single lever
2. **3-layer depth recurrence** — 11 physical layers → 17 virtual layers (layers 3-5 loop 3×). Massive param saving.
3. **Parallel residuals** (GPT-J style) — attention and MLP read from same pre-residual input, outputs summed. Different from sequential baseline.
4. **QK-gain 5.25** — baseline is 1.5, pushed aggressively. Monotonic improvement from 4.0→5.25.
5. **Legal TTT** — at eval time, keep training on already-scored val tokens (SGD, 3 epochs per 32K chunk). ~0.002-0.003 bpb gain.
6. **INT6 quantization** — more aggressive than baseline INT8.
7. **160+ experiments** on OpenAI compute grant ($500 RunPod credits)

**Key lesson:** gains are stacked. No single trick gets you from 1.2 to 1.08. Each piece contributes 0.01-0.03 bpb.

---

## Session takeaways — the ~8 substantive diffs from nanoGPT

| Axis | nanoGPT | parameter-golf | Why for PG |
|---|---|---|---|
| Normalization | LayerNorm (γ,β params) | RMSNorm (0 params) | Save ~38K params |
| Position encoding | learned wpe (786K params) | Rotary (0 params, no length limit) | Save params + length generalization |
| Linear layer | nn.Linear bf16 | CastedLinear (fp32 stored, bf16 compute) | Training speed + optimizer stability |
| Attention heads | equal Q,K,V heads | GQA — fewer K,V heads | Smaller K,V matrices + cache |
| Attention stability | none | QK-norm + q_gain | Stable training at high LR |
| Attention compute | manual softmax(QKᵀ/√d)V | F.scaled_dot_product_attention | FlashAttention 2-4× faster |
| MLP activation | GELU | ReLU² | Sparsity + speed |
| Residual structure | sequential | U-Net skips + x0 | Gradient flow, expressiveness |
| Optimizer | AdamW | Muon (orthogonalized gradients) | Better matrix optimization |
| Artifact format | full precision | INT8 + zlib | 4× compression → fits 16 MB |
| Tokenizer | GPT-2 BPE fixed | SentencePiece pluggable | Vocab size as tunable lever |

## What this teaches me about what to try

See `projects/parameter-golf/ideas.md` for ranked experiment directions. Top picks:
1. **Depth recurrence** — highest param-efficiency gain, proven on leaderboard
2. **SP4096** — clean one-config-change experiment, well-proven
3. **INT4/INT6 quantization** — doubles effective param budget
4. **Lower mlp_mult** (try M=1) — halves MLP cost, more room for layers
5. **q_gain tuning** — free, monotonic improvement observed on leaderboard
