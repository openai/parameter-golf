# HELIX Architecture Design Spec

> **Hypernetwork-assisted Efficient Looped Information eXchange**
> Parameter Golf Challenge | 16MB artifact | 8×H100 | ≤10 min
> Target: Beat current SOTA BPB of **1.1194** (2026-03-23)

---

## 1. Goal

Implement HELIX — a novel transformer architecture combining three cutting-edge 2025 research innovations never previously combined or applied to this competition:

1. **D-TPA** — Differential Tensor Product Attention: outer-product factored Q/K/V (TPA, ICML 2025) with noise-canceling differential attention maps (DIFF, ICLR 2025 Oral)
2. **MoR** — Mixture of Recursions: 5 shared blocks × 3 max iterations, where each token decides per-round whether to exit early or continue (arXiv:2507.10524, July 2025)
3. **Peri-LN** — Sandwich normalization (pre-LN + post-LN at every sublayer, ICML 2025, adopted by Gemma 2 / OLMo 2)

These combine with the proven SOTA support stack: SmearGate, BigramHash, int6+zstd-22, partial RoPE, XSA on deep blocks, EMA(0.997), SWA, Muon optimizer.

**Projected BPB:** 1.097–1.107 (vs SOTA 1.1194).

---

## 2. Architecture

### 2.1 High-Level Structure

```
input_ids [B, T]
    ↓ tok_emb(1024, 768) + bigram(2048, 128→768)
    ↓ RMSNorm → SmearGate
    ↓ x0 = x  (anchor for resid_mix)
    ↓
[Iteration r=0]
  Block_0(x, x0, r=0) → Block_1 → Block_2 → Block_3 → Block_4
  [collect U-Net skip from Block_0, Block_1 outputs]
  MoR gate_0(x): compute exit logit, store for aux loss (all tokens continue)
    ↓
[Iteration r=1]
  Block_0(x, x0, r=1) → ... → Block_4
  MoR gate_1(x): compute exit logit, store for aux loss (all tokens continue)
    ↓
[Iteration r=2]
  [inject U-Net skips (reversed) into Block_0, Block_1]
  Block_0(x, x0, r=2) → ... → Block_4
    ↓
RMSNorm(x) → lm_head (tied to tok_emb.T) → softcap → logits
```

**MoR at training time:** All tokens run all 3 iterations (soft gating). Gates are trained via load-balancing auxiliary loss. **MoR at eval time:** All 3 iterations always run (gates ignored). This maximizes BPB quality.

### 2.2 Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| `NUM_UNIQUE_BLOCKS` (K) | 5 | 5×3=15 virtual layers > 11 SOTA layers |
| `NUM_ITERATIONS` (R) | 3 | Matches Looped Transformer "≥L/3 unique blocks" theorem |
| `MODEL_DIM` (d) | 768 | D-TPA saves ~77% attn params → reinvest in width |
| `NUM_HEADS` | 8 | d_head = 768/8 = 96; DIFF splits to 48 per sub-head |
| `NUM_KV_HEADS` | 4 | GQA 8Q/4KV ratio maintained |
| `DTPA_RANK` (r) | 4 | Rank-4 outer-product factorization per head |
| `FFN_HIDDEN` | 1536 | SwiGLU 2× expansion (hidden=2d). **Isoparametric with relu²(3d)**: SwiGLU(2d) uses 3×d×2d=6d² params = relu²(3d)'s 2×d×3d=6d². Same parameter budget, better activation. |
| `ROPE_DIMS` | 16 | Partial RoPE: 16 of 48 sub-head dims per DIFF pair |
| `XSA_LAST_N` | 2 | XSA on blocks 3,4 at iteration r=2 only |
| `MOR_LB_WEIGHT` | 0.01 | Load-balancing aux loss weight |
| `MOR_LB_DECAY_STEPS` | 1000 | Steps over which lb_weight decays to 0 during warmdown |
| `BIGRAM_VOCAB_SIZE` | 2048 | BigramHash buckets |
| `BIGRAM_DIM` | 128 | BigramHash projection dim |

### 2.3 Parameter Budget

| Component | Params |
|---|---|
| D-TPA per block (d=768, rank=4) | ~399K |
| SwiGLU FFN per block (d=768, hidden=1536) | ~3,539K |
| Peri-LN (4 RMSNorm per block, d=768) | ~3K |
| Per-depth scales + MoR router per block | ~11K |
| **Per-block subtotal** | **~3,952K** |
| 5 unique blocks | ~19,760K |
| tok_emb (1024×768) | 786K |
| BigramHash (2048×128 + 128×768) | 361K |
| SmearGate gate (768) + skip_weights (2×768) + misc | ~2K |
| **Grand total** | **~20,909K ≈ 20.91M** |
| Artifact @ 0.726 B/param (int6+zstd-22) | **~15.18MB ✓** |

Headroom: **0.82MB** below 16MB limit.

---

## 3. D-TPA: Differential Tensor Product Attention

### 3.1 Motivation

Two independent 2025 papers each improve attention in orthogonal ways:
- **TPA** reduces Q/K/V parameter count ~77% via outer-product factorization while maintaining or improving quality. Natively integrates RoPE (applied to reconstructed Q/K tensors after factored reconstruction, per-position, per-forward-pass).
- **DIFF** cancels attention noise by subtracting two attention maps. Proven to reduce required training compute by ~35% to reach equivalent quality.

D-TPA combines them: TPA provides efficient factored Q/K/V; DIFF provides cleaner attention patterns. These are orthogonal improvements.

### 3.2 Computation

```python
# d=768, n_heads=8, n_kv=4, rank r=4, d_head=96, d_head_half=48
# DIFF: each attention "head" uses two sub-heads of d_head//2=48 dims each

# --- Step 1: Factored Q/K/V reconstruction ---
# W_cQ: [d, 2*n_heads*rank] = [768, 64]  (2 for DIFF pairs, 8 heads, rank 4)
# A_Q:  [2*n_heads, rank, d_head//2]     = [16, 4, 48]  (static basis, trained)
c_Q = x @ W_cQ                            # [B, T, 64]
c_Q = c_Q.view(B, T, 2*n_heads, rank)    # [B, T, 16, 4]
# einsum: sum over rank dimension
Q_all = (c_Q.unsqueeze(-1) * A_Q.unsqueeze(0).unsqueeze(0)).sum(-2)
# Q_all: [B, T, 2*n_heads, d_head//2]
Q1, Q2 = Q_all.chunk(2, dim=2)           # each [B, T, n_heads=8, 48]

# Similarly reconstruct K1, K2 [B, T, n_kv=4, 48] and V1, V2 [B, T, n_kv=4, 48]
# using W_cK [768, 32], A_K [8, 4, 48]  and  W_cV [768, 32], A_V [8, 4, 48]

# --- Step 2: Apply partial RoPE to reconstructed Q/K ---
# RoPE is applied per-position, per-forward-pass, AFTER factored reconstruction.
# apply_rotary_emb operates on [B, heads, T, d_head//2]; transpose before/after.
Q1 = apply_rotary_emb(Q1.transpose(1,2), cos, sin, rope_dims=ROPE_DIMS).transpose(1,2)
Q2 = apply_rotary_emb(Q2.transpose(1,2), cos, sin, rope_dims=ROPE_DIMS).transpose(1,2)
K1 = apply_rotary_emb(K1.transpose(1,2), cos, sin, rope_dims=ROPE_DIMS).transpose(1,2)
K2 = apply_rotary_emb(K2.transpose(1,2), cos, sin, rope_dims=ROPE_DIMS).transpose(1,2)

# --- Step 3: GQA expansion and QK-norm ---
# Expand KV from n_kv=4 heads to n_heads=8 (each KV head serves 2 Q heads)
K1 = K1.repeat_interleave(n_heads // n_kv, dim=2)  # [B, T, 8, 48]
K2 = K2.repeat_interleave(n_heads // n_kv, dim=2)
V1 = V1.repeat_interleave(n_heads // n_kv, dim=2)
V2 = V2.repeat_interleave(n_heads // n_kv, dim=2)
# QK-norm: normalize Q and K before attention
Q1, Q2 = rms_norm(Q1), rms_norm(Q2)
K1, K2 = rms_norm(K1), rms_norm(K2)

# --- Step 4: Differential attention ---
# lam: [n_heads=8], per-Q-head scalar (shape matches n_heads after GQA expansion)
# Init: lam[h] = 0.8 - 0.6 * exp(-0.3 * (layer_idx - 1)) per block
scale = 1.0 / math.sqrt(d_head // 2)  # = 1/sqrt(48)
A1 = F.scaled_dot_product_attention(Q1, K1, V1, scale=scale, is_causal=True)
A2 = F.scaled_dot_product_attention(Q2, K2, V2, scale=scale, is_causal=True)
# lam broadcast: [1, 1, n_heads, 1] to match [B, T, n_heads, d_head//2]
out = A1 - lam.view(1, 1, n_heads, 1) * A2               # [B, T, 8, 48]

# GroupNorm after differential subtraction for numerical stability
# (follows the original DIFF paper's recommendation)
out = F.group_norm(out.reshape(B, T, -1), num_groups=n_heads).reshape(B, T, n_heads, 48)

# --- Step 5: Output projection ---
out = out.reshape(B, T, n_heads * d_head // 2)  # [B, T, 384]
out = out @ W_O                                  # [B, T, 768]  W_O: [384, 768]
```

**Note on `F.scaled_dot_product_attention` with DIFF:** The differential subtraction happens on the weighted-sum *outputs* (after attention × V), not on the attention maps directly. This is equivalent to the DIFF formulation `(A1 - λA2)V = A1V - λA2V` by linearity. This form is compatible with FlashAttention-3 (FA3) since each `scaled_dot_product_attention` call runs as a standard attention kernel. **GroupNorm** (num_groups=n_heads) is applied to the concatenated output before `W_O`, providing the normalization recommended in the DIFF paper.

### 3.3 Parameter Count Detail

| Tensor | Shape | Params |
|---|---|---|
| W_cQ | 768 × (2×8×4) = 768×64 | 49,152 |
| A_Q (basis) | 2×8×4×48 = 16×4×48 | 3,072 |
| W_cK | 768 × (2×4×4) = 768×32 | 24,576 |
| A_K (basis) | 2×4×4×48 = 8×4×48 | 1,536 |
| W_cV | 768 × 32 | 24,576 |
| A_V (basis) | 8×4×48 | 1,536 |
| W_O | 384 × 768 | 294,912 |
| λ (per Q-head after expansion) | 8 | 8 |
| q_gain (per Q-head) | 8 | 8 |
| **Total D-TPA** | | **~399,376 ≈ 399K** |

Compare: standard GQA at d=768 costs `768×768 + 768×384×2 + 768×768 = 2,359K`. D-TPA costs **399K — 83% reduction** in attention parameters.

### 3.4 XSA Integration with D-TPA

XSA (Exclusive Self-Attention) is applied to the last 2 blocks (k=3, k=4) at iteration r=2 only. Since the differential output `A1V - λA2V` is already processed by GroupNorm before W_O, XSA replaces the **V computation** at this point: instead of using learned V projections, the block's residual stream `x` acts as the value signal.

```python
if self.use_xsa and r == self.num_iterations - 1:
    # XSA: re-run attention using x as V, no V projection
    # Use Q1 and K1 only (one clean attention map, no differential for XSA)
    A_xsa = F.scaled_dot_product_attention(Q1, K1, x_as_v, scale=scale, is_causal=True)
    # x_as_v: x transposed to [B, n_heads, T, d//n_heads] by tiling across kv heads
    out = A_xsa.reshape(B, T, n_heads * (d // n_heads))
    out = out @ W_O_xsa  # separate W_O for XSA blocks: [(d//n_heads)*n_heads, d]
```

XSA blocks have a second output projection `W_O_xsa` (same shape as W_O). These are initialized separately and contribute ~295K params to the 2 XSA-enabled blocks (additional ~590K total, absorbed in the ~0.82MB headroom).

### 3.5 Optimizer Routing for D-TPA

The basis tensors `A_Q`, `A_K`, `A_V` are **3D** (`[heads, rank, d_head_half]`). Muon requires 2D matrices. These must be explicitly routed to AdamW. They also should not be treated as scalar control parameters. Route them to a separate Adam group with `lr=SCALAR_LR`.

The `mor_gate` parameters (`[d, 1]`, ndim=2) must be routed to **scalar AdamW**, not Muon. Ensure that `"mor_gate"` is added to `CONTROL_TENSOR_NAME_PATTERNS` (see Section 6.3).

---

## 4. MoR: Mixture of Recursions

### 4.1 Forward Pass Structure

```python
def forward(self, input_ids, target_ids):
    x = self._embed(input_ids)   # embed + bigram + smear
    x0 = x.clone()
    first_iter_hidden = []
    mor_gate_logits = []         # accumulated per-iteration for aux loss

    for r in range(R_MAX):
        for k in range(K):
            # U-Net: inject encoder skips into last iteration (no detach — full BPTT)
            if r == R_MAX - 1 and k < self.num_skip:
                skip_idx = self.num_skip - 1 - k
                w = self.skip_weights[skip_idx].to(dtype=x.dtype)
                x = x + w[None, None, :] * first_iter_hidden[skip_idx]
            x = self.blocks[k](x, x0, r)
            # U-Net: collect first-iteration hiddens (no detach)
            if r == 0 and k < self.num_skip:
                first_iter_hidden.append(x)

        # MoR gate (between iterations, not between blocks)
        if r < R_MAX - 1:
            gate_logit = x @ self.mor_gate[r]   # [B, T, 1]
            mor_gate_logits.append(gate_logit)
            # Training: all tokens continue (soft gating, no masking)
            # Inference: same — all tokens always run all 3 iterations

    x = self.final_norm(x)

    # Compute cross-entropy loss
    logits = self._project_logits(x)
    ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE).float(),
                              target_ids.reshape(-1))

    # MoR auxiliary load-balancing loss (added to ce_loss)
    aux_loss = self._mor_aux_loss(mor_gate_logits)
    return ce_loss + aux_loss
```

**Why no `.detach()` on U-Net skips:** Gradients must flow through all paths for correct BPTT. The original code comment "avoid gradient loops" was incorrect — there are no gradient loops here since the computation graph is acyclic (each iteration only reads `first_iter_hidden`, which was written in a previous iteration).

### 4.2 MoR Router Parameters

```python
self.mor_gate = nn.ParameterList([
    nn.Parameter(torch.zeros(d, 1))   # gate vector per inter-iteration boundary
    for _ in range(R_MAX - 1)         # 2 gates total
])
# Total: 2 × 768 = 1,536 params
# Named "mor_gate.*" → must be in CONTROL_TENSOR_NAME_PATTERNS to route to AdamW
```

### 4.3 Load-Balancing Auxiliary Loss

Under soft routing (all tokens run all iterations at training time), both gates see all tokens. The load-balancing target is that each gate would, if used as a hard router, send ~33% of tokens out at each iteration.

```python
def _mor_aux_loss(self, gate_logits):
    # gate_logits: list of 2 tensors of shape [B, T, 1]
    # p[i] = mean probability of exiting at boundary i
    p0 = torch.sigmoid(gate_logits[0]).mean()    # exit rate at iter 0→1
    p1 = torch.sigmoid(gate_logits[1]).mean()    # exit rate at iter 1→2
    # Target: uniform distribution across 3 exit points
    target = 1.0 / 3.0
    lb_weight = self._current_lb_weight          # set by training loop, decays to 0
    aux = lb_weight * ((p0 - target)**2 + (p1 - target)**2)
    return aux
```

`_current_lb_weight` is updated by the training loop scheduler (starts at `MOR_LB_WEIGHT=0.01`, linearly decays to 0 over the last `MOR_LB_DECAY_STEPS=1000` warmdown steps).

**Integration:** `forward()` returns `ce_loss + aux_loss` directly. No training loop modification needed — the returned scalar includes both losses. The training loop calls `loss.backward()` on this scalar as usual.

### 4.4 `torch.compile` Compatibility

The MoR loop uses Python-level `for r, for k` loops and a Python list `first_iter_hidden`. These are **incompatible with `fullgraph=True`** compilation. The training script must use `fullgraph=False`:

```python
# In training loop (replaces the existing compile line):
base_model = torch.compile(base_model, dynamic=False, fullgraph=False)
```

`fullgraph=False` still applies kernel-level compilation but allows Python control flow. Performance impact: approximately 10–20% slower than `fullgraph=True`, but the 15 effective layers (vs 11 in SOTA) more than compensate in training throughput. The recursive structure also means more compute per forward pass, so the relative overhead of Python loop interpretation is smaller.

---

## 5. Per-Depth Adaptation and Block Structure

### 5.1 Peri-LN (Sandwich Normalization)

Each sublayer (attention and MLP) gets pre-norm AND post-norm:

```python
# Attention sublayer
h = self.pre_norm_attn(x)          # Pre-LN (RMSNorm)
h = self.dtpa(h, r=r)
h = self.post_norm_attn(h)         # Post-LN (RMSNorm) ← Peri-LN addition
x = x + self.iter_attn_scale[r] * h

# MLP sublayer
h = self.pre_norm_mlp(x)           # Pre-LN (RMSNorm)
h = self.swiglu(h)
h = self.post_norm_mlp(h)          # Post-LN (RMSNorm) ← Peri-LN addition
x = x + self.iter_mlp_scale[r] * h
```

4 RMSNorm instances per block × 5 blocks = 20 total. Cost: 20 × 768 = 15,360 params. Negligible.

### 5.2 Per-Iteration Scalars

```python
# Per block: 3 ParameterLists, each with R=3 entries
# Named to match CONTROL_TENSOR_NAME_PATTERNS → float32 AdamW

self.iter_attn_scale = nn.ParameterList([
    nn.Parameter(torch.ones(d))  for _ in range(R)   # "attn_scale" → control
])
self.iter_mlp_scale = nn.ParameterList([
    nn.Parameter(torch.ones(d))  for _ in range(R)   # "mlp_scale" → control
])
self.iter_resid_mix = nn.ParameterList([
    nn.Parameter(torch.stack([torch.ones(d), torch.zeros(d)]))
    for _ in range(R)                                 # "resid_mix" → control
])
# Per-block cost: 3 × (768 + 768 + 2×768) = 9,216 params
```

### 5.3 SwiGLU FFN

```python
class SwiGLU(nn.Module):
    """
    SwiGLU with hidden = 2 * dim.
    Isoparametric to relu²(3 * dim): both use 6d² weight entries.
      SwiGLU(h=2d): gate(d→2d) + fc(d→2d) + proj(2d→d) = 3 × d × 2d = 6d²
      relu²(h=3d):  fc1(d→3d) + fc2(3d→d)              = 2 × d × 3d = 6d²
    SwiGLU advantage: smooth gradient, multiplicative gating.
    At d=768, hidden=1536: 3 × 768 × 1536 = 3,538,944 params (same as relu²(2304))
    """
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = CastedLinear(dim, hidden)
        self.fc   = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.fc(x))
```

### 5.4 Full Block Forward

```python
class HELIXBlock(nn.Module):
    def forward(self, x, x0, r):
        # Residual mixing (iter-specific blend of current state and initial anchor)
        mix = self.iter_resid_mix[r].to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention sublayer with Peri-LN
        h = self.pre_norm_attn(x)
        h = self.dtpa(h, layer_r=r)        # D-TPA (handles XSA flag internally)
        h = self.post_norm_attn(h)
        x = x + self.iter_attn_scale[r].to(dtype=x.dtype)[None, None, :] * h

        # MLP sublayer with Peri-LN
        h = self.pre_norm_mlp(x)
        h = self.swiglu(h)
        h = self.post_norm_mlp(h)
        x = x + self.iter_mlp_scale[r].to(dtype=x.dtype)[None, None, :] * h
        return x
```

---

## 6. Quantization (int6+zstd-22)

The int6+zstd-22 quantization pipeline is **not in the baseline `train_gpt.py`**. It comes from the SOTA submission at `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`. The implementation plan must:
1. Copy the `quantize_model_int6()` and `save_artifact_zstd()` functions from the SOTA submission into the new `train_gpt.py`
2. Verify these functions operate on flat 2D weight tensors — D-TPA introduces 3D basis tensors (`A_Q`, `A_K`, `A_V`) which must be reshaped to 2D before quantization (e.g., `A_Q.reshape(2*n_heads*rank, d_head_half)`) and reshaped back at load time
3. The 0.726 bytes/param estimate already accounts for the int6 compression ratio; the 3D reshape does not change the total parameter count

---

## 7. Training Configuration

### 7.1 Optimizer Setup

```python
CONTROL_PATTERNS = ('attn_scale', 'mlp_scale', 'resid_mix',
                    'q_gain', 'skip_weight', 'smear', 'mor_gate')

# Matrix params → Muon (2D, not matching control patterns)
muon_params = [p for n, p in base_model.blocks.named_parameters()
               if p.ndim == 2
               and not any(pat in n for pat in CONTROL_PATTERNS)]

# 3D TPA basis tensors → separate Adam group (cannot use Muon on 3D)
dtpa_basis_params = [p for n, p in base_model.named_parameters()
                     if 'A_q' in n or 'A_k' in n or 'A_v' in n]

# Scalar/control → float32 AdamW
scalar_params = [p for n, p in base_model.named_parameters()
                 if any(pat in n for pat in CONTROL_PATTERNS)]
```

### 7.2 Learning Rates

| Parameter Group | LR | Rationale |
|---|---|---|
| Matrix params (Muon) | 0.023 | `0.04 / √3` — 3× gradient accumulation from 3 shared-weight iterations |
| Scalar/control (Adam) | 0.04 | Standard |
| Embeddings (Adam) | 0.05 | Standard |
| TPA basis A_Q/K/V (Adam) | 0.04 | 3D tensors; role analogous to projection matrices |

### 7.3 Full Environment Variables

```bash
RUN_ID=helix_v1 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
DTPA_RANK=4 \
NUM_UNIQUE_BLOCKS=5 \
NUM_ITERATIONS=3 \
ROPE_DIMS=16 \
XSA_LAST_N=2 \
FFN_HIDDEN=1536 \
MOR_LB_WEIGHT=0.01 \
MOR_LB_DECAY_STEPS=1000 \
TRAIN_SEQ_LEN=2048 \
WARMDOWN_ITERS=3500 \
GRAD_CLIP_NORM=0.3 \
MATRIX_LR=0.023 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MUON_WD=0.04 \
ADAM_WD=0.01 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=1 \
SWA_EVERY=200 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
LOGIT_SOFTCAP=30.0 \
CONTROL_TENSOR_NAME_PATTERNS="attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,smear,mor_gate" \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 7.4 Initialization

```python
def _init_weights(self):
    virtual_depth = self.num_unique_blocks * self.num_iterations  # 5*3=15
    output_scale = 1.0 / math.sqrt(2 * virtual_depth)            # 1/sqrt(30)

    # Linear weights: orthogonal init + muP output scaling
    for name, module in self.named_modules():
        if isinstance(module, CastedLinear):
            if getattr(module, '_zero_init', False):
                nn.init.zeros_(module.weight)
            else:
                nn.init.orthogonal_(module.weight)
                module.weight.data.mul_(output_scale)

    # TPA basis tensors (3D): orthogonal init per head-slice
    for name, param in self.named_parameters():
        if any(x in name for x in ('A_q', 'A_k', 'A_v')):
            # param: [2*n_heads, rank, d_head_half] → init each head's rank×d_half slice
            n_slices = param.shape[0]
            for i in range(n_slices):
                nn.init.orthogonal_(param.data[i])  # [rank, d_head_half]

    # Lambda (DIFF): per-block initialization
    for block_idx, block in enumerate(self.blocks):
        lam_val = 0.8 - 0.6 * math.exp(-0.3 * block_idx)
        block.dtpa.lam.data.fill_(lam_val)

    # Embeddings
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, 0.0, self.tied_embed_init_std)
```

---

## 8. Interface Contract

HELIX exposes the standard SOTA training-loop interface:

```python
base_model.blocks           # nn.ModuleList of 5 HELIXBlock instances
base_model.smear            # SmearGate (has .gate attribute)
base_model.bigram           # BigramHashEmbedding(.embed.weight, .proj, .scale)
base_model.tok_emb          # nn.Embedding (tied with lm_head)
base_model.skip_weights     # nn.Parameter [2, 768]
base_model.mtp_heads        # nn.ModuleList([]) — unused
base_model.mtp_num_heads    # 0
base_model.forward_logits(input_ids)  # [B, T, V] — all 3 iters, MoR gates inactive
```

`forward(input_ids, target_ids)` returns `ce_loss + aux_loss` as a single scalar. No training loop modification required.

---

## 9. File Structure

| File | Role |
|---|---|
| `train_gpt.py` | HELIX model replaces `Block` + `GPT` classes (~250 lines); int6+zstd-22 quantization from SOTA submission copied in; `fullgraph=False` for compile; `CONTROL_TENSOR_NAME_PATTERNS` updated; optimizer routing for 3D TPA basis added |
| `train_gpt_mlx.py` | MLX smoke-test mirror (simplified DTPA, no MoR) |

**Line budget:**
- SOTA submission base: ~1480 lines
- Remove `Block` + `GPT`: −250 lines
- Add `DTPA`, `SwiGLU`, `HELIXBlock`, `HELIX_GPT`: +350 lines
- Minor changes (optimizer, env vars, compile flag): +30 lines
- **Net: ~1610 lines** — exceeds 1500-line hard limit by ~110 lines

**Mitigation:** Consolidate `DTPA` into `HELIXBlock` (no separate class, inline the forward), remove the `count_parameters` utility, and use `# fmt: skip` to condense multi-line parameter definitions. Target: stay within 1500 lines.

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| D-TPA not proven at <25M scale | Medium | TPA and DIFF each proven separately; combination is modular |
| `fullgraph=False` compile slows training by 15-20% | Medium | 15 virtual layers provide more computation per step; net throughput similar |
| Recursive convergence slower than independent layers | Medium | Orthogonal init + muP scaling + per-depth scales |
| MoR gate collapse (all tokens exit early) | Low-Medium | Load-balancing aux loss; decay ensures it doesn't over-constrain |
| 3D TPA basis tensors excluded from optimizer | Low | Explicit routing in optimizer setup (Section 7.1) |
| Artifact size exceeds 16MB | Low | 0.82MB headroom; XSA-extra W_O adds ~590K × 0.726 = 0.43MB, leaving 0.39MB |

---

## 11. Expected Performance

| Source of gain | Expected ΔBPB |
|---|---|
| D-TPA (more expressive attention per param) | −0.005 to −0.012 |
| Peri-LN (sandwich norm) | −0.002 to −0.004 |
| Wider d=768 vs 512 (freed attn param budget) | −0.005 to −0.010 |
| MoR (forces iteration-depth specialization during training) | −0.002 to −0.005 |
| 15 virtual layers vs 11 independent | −0.003 to +0.003 (uncertain) |
| **Combined** | **−0.012 to −0.022** |

**Projected BPB: 1.097–1.107** vs current SOTA 1.1194.

---

## 12. References

1. **TPA**: "Tensor Product Attention Is All You Need" — arXiv:2501.06425 (Jan 2025, ICML 2025)
2. **Differential Attention**: "Differential Transformer" — arXiv:2410.05258 (Oct 2024, ICLR 2025 Oral)
3. **Peri-LN**: "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture" — arXiv:2502.02732 (Feb 2025, ICML 2025)
4. **MoR**: "Mixture of Recursions" — arXiv:2507.10524 (July 2025)
5. **SwiGLU**: Shazeer 2020 / LLaMA — isoparametric with relu² at hidden=2d
6. **XSA**: "Exclusive Self-Attention" — arXiv:2603.09078 (2026)
7. **Looped Transformers**: Giannou et al. 2024 — theoretical foundation for recursive depth
8. **Universal Transformers**: Dehghani et al. 2018 — empirical foundation
9. **SmearGate / BigramHash**: parameter-golf community (PR #414 stack)
10. **DIFF GroupNorm**: "Differential Transformer" Section 2.2 — applied post-differential for stability
