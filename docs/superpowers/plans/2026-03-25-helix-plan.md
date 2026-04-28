# HELIX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the HELIX architecture (D-TPA + MoR + Peri-LN) as a new competitive submission targeting BPB ≤ 1.107 vs current SOTA 1.1194.

**Architecture:** 5 unique HELIXBlocks × 3 MoR iterations (15 virtual layers), d=768, D-TPA attention (rank-4 factored Q/K/V with differential subtraction), SwiGLU FFN, Peri-LN (sandwich norm), U-Net skip connections, MoR load-balancing loss.

**Tech Stack:** PyTorch, CUDA/DDP-free (Parallel Muon handles grad sync), flash_attn_interface (FA3), int6+lzma quantization (SOTA uses `lzma.compress(preset=6)`, despite CLAUDE.md saying zstd-22), SentencePiece tokenizer.

**Spec:** `docs/superpowers/specs/2026-03-25-helix-design.md`
**Base file:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` (SOTA, 1898 lines)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py` | Create | Full HELIX training script (built on SOTA base) |
| `records/track_10min_16mb/2026-03-25_HELIX/submission.json` | Create | Submission metadata (filled after training) |
| `records/track_10min_16mb/2026-03-25_HELIX/README.md` | Create | Submission description |

**What is kept from SOTA base (do NOT rewrite these):**
- Lines 1–100: imports, Hyperparameters class skeleton
- Lines 102–360: Parallel Muon optimizer (`zeropower_via_newtonschulz5`, `Muon` class)
- Lines 360–430: Quantization helpers (`CONTROL_TENSOR_NAME_PATTERNS`, `keep_float_tensor`, `quantize_float_tensor`, `quantize_state_dict_int8`)
- Lines 430–674: `DistributedTokenLoader`, BPB eval helpers, `build_sentencepiece_luts`, `eval_val`, `eval_val_sliding`
- Lines 675–706: `SmearGate`, `BigramHashEmbedding`
- Lines 1230–1380: `quantize_int6_per_row`, `mixed_quantize_int6`, `dequantize_mixed_int6`
- Lines 1381+: `main()` skeleton (DDP setup, logging, EMA, SWA, training loop)

**What is replaced:**
- SOTA `Hyperparameters` fields: add HELIX-specific fields
- SOTA `CONTROL_TENSOR_NAME_PATTERNS` default string: add `"mor_gate"`
- SOTA `MLP`, `Block`, `GPT` classes (lines 724–1229): replace with `SwiGLU`, `DTPA`, `HELIXBlock`, `HELIX_GPT`
- SOTA `_classify_param`: add `.dtpa.` and `.swiglu.` rules
- SOTA `main()`: swap `GPT(...)` → `HELIX_GPT(...)`, update optimizer routing, change `fullgraph=True` → `fullgraph=False`, remove bank tensor setup, add MoR lb_weight decay
- SOTA serialization: remove `_unbank_state_dict`/`_rebank_state_dict` calls, update eval_model instantiation

---

## Task 1: Bootstrap — Create Submission Directory and Copy Base

**Files:**
- Create: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py`

- [ ] **Step 1: Copy SOTA submission as starting point**

```bash
cp records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py \
   records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
```

- [ ] **Step 2: Verify copy succeeded**

```bash
wc -l records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
```
Expected: `1898 records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py`

- [ ] **Step 3: Commit the bootstrap**

```bash
git add records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
git commit -m "feat: bootstrap HELIX submission from SOTA base"
```

---

## Task 2: Update Hyperparameters and CONTROL_TENSOR_NAME_PATTERNS

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py` (lines ~28–101 and ~363–370)

**What to add to `Hyperparameters` class** (after existing fields, before the closing):

```python
    # --- HELIX architecture fields ---
    dtpa_rank = int(os.environ.get("DTPA_RANK", 4))
    num_unique_blocks = int(os.environ.get("NUM_UNIQUE_BLOCKS", 5))
    num_iterations = int(os.environ.get("NUM_ITERATIONS", 3))
    ffn_hidden = int(os.environ.get("FFN_HIDDEN", 1536))
    # HELIX overrides: d=768, 8Q/4KV, rope_dims=16, xsa_last_n=2
    # (override existing SOTA defaults in the same fields)
    # MoR load-balancing
    mor_lb_weight = float(os.environ.get("MOR_LB_WEIGHT", 0.01))
    mor_lb_decay_steps = int(os.environ.get("MOR_LB_DECAY_STEPS", 1000))
```

**Also update these EXISTING Hyperparameter defaults** (change the default values in os.environ.get):

| Field | SOTA default | HELIX default |
|---|---|---|
| `model_dim` | `512` | `768` |
| `num_layers` | `11` | `5` (num_unique_blocks; loops handle virtual depth) |
| `rope_dims` | `16` | `16` (same, but document it's for d_head_half=48) |
| `xsa_last_n` | `4` | `2` |
| `matrix_lr` | `0.025` | `0.023` (=0.04/√3 for 3× gradient accumulation) |
| `warmdown_iters` | `3500` | `3500` (keep) |

**Update `CONTROL_TENSOR_NAME_PATTERNS` default string** — add `"mor_gate"`:

```python
# Find this line (around line 363):
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
    ).split(",")
    if pattern
)
# Change to:
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda,mor_gate",
    ).split(",")
    if pattern
)
```

- [ ] **Step 1: Write a minimal smoke test**

Create `records/track_10min_16mb/2026-03-25_HELIX/test_hparams.py`:

```python
"""Sanity-check that Hyperparameters loads without error and fields exist."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
# Avoid importing CUDA-dependent modules by patching flash_attn_interface
import types, unittest.mock
sys.modules.setdefault('flash_attn_interface', types.SimpleNamespace(flash_attn_func=None))
import train_gpt as tg
args = tg.Hyperparameters()
assert hasattr(args, 'dtpa_rank'), "dtpa_rank missing"
assert hasattr(args, 'num_unique_blocks'), "num_unique_blocks missing"
assert hasattr(args, 'num_iterations'), "num_iterations missing"
assert hasattr(args, 'ffn_hidden'), "ffn_hidden missing"
assert hasattr(args, 'mor_lb_weight'), "mor_lb_weight missing"
assert hasattr(args, 'mor_lb_decay_steps'), "mor_lb_decay_steps missing"
assert args.dtpa_rank == 4
assert args.num_unique_blocks == 5
assert args.model_dim == 768
assert "mor_gate" in tg.CONTROL_TENSOR_NAME_PATTERNS
print("PASS: Hyperparameters OK")
```

- [ ] **Step 2: Run test to verify it FAILS (missing fields)**

```bash
cd records/track_10min_16mb/2026-03-25_HELIX
python test_hparams.py
```
Expected: `AssertionError: dtpa_rank missing`

- [ ] **Step 3: Implement the hyperparameter changes** (edit `train_gpt.py` as described above)

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_hparams.py
```
Expected: `PASS: Hyperparameters OK`

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py \
        records/track_10min_16mb/2026-03-25_HELIX/test_hparams.py
git commit -m "feat(helix): add HELIX hyperparameters and mor_gate control pattern"
```

---

## Task 3: Implement SwiGLU FFN

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py`
- Insert after `BigramHashEmbedding` class (around line 706), replacing SOTA's `MLP` class

**Replace the SOTA `MLP` class with `SwiGLU`:**

```python
class SwiGLU(nn.Module):
    """
    SwiGLU FFN with hidden = 2 * dim.
    Isoparametric to relu²(3*dim): both use 6d² weight entries.
      SwiGLU(h=2d): gate(d→2d) + fc(d→2d) + proj(2d→d) = 3×d×2d = 6d²
      relu²(h=3d): fc1(d→3d) + fc2(3d→d)               = 2×d×3d = 6d²
    At d=768, hidden=1536: 3×768×1536 = 3,538,944 params per block.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.fc(x))
```

- [ ] **Step 1: Write failing test**

Add to `test_hparams.py` (or create `test_swiglu.py`):

```python
"""Test SwiGLU forward pass shapes and param count."""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault('flash_attn_interface', types.SimpleNamespace(flash_attn_func=None))
import torch
import train_gpt as tg

B, T, D, H = 2, 16, 768, 1536
swiglu = tg.SwiGLU(D, H)
x = torch.randn(B, T, D)
out = swiglu(x)
assert out.shape == (B, T, D), f"Expected ({B},{T},{D}), got {out.shape}"
n_params = sum(p.numel() for p in swiglu.parameters())
assert n_params == 3 * D * H, f"Expected {3*D*H} params, got {n_params}"
print(f"PASS: SwiGLU OK  params={n_params:,}")
```

- [ ] **Step 2: Run test to verify it FAILS**

```bash
python test_swiglu.py
```
Expected: `AttributeError: module 'train_gpt' has no attribute 'SwiGLU'`

- [ ] **Step 3: Implement `SwiGLU`** in `train_gpt.py` (replace `MLP` class as shown above)

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_swiglu.py
```
Expected: `PASS: SwiGLU OK  params=3,538,944`

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/
git commit -m "feat(helix): add SwiGLU FFN (isoparametric with relu² at hidden=2d)"
```

---

## Task 4: Implement DTPA (Differential Tensor Product Attention)

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py`
- Insert after `SwiGLU`, replacing SOTA's `CausalSelfAttention` class

**Full `DTPA` implementation:**

```python
class DTPA(nn.Module):
    """
    Differential Tensor Product Attention.
    Combines TPA (factored Q/K/V via rank-r outer products) with
    Differential Attention (noise-canceling A1 - λ*A2).

    Parameters per block at d=768, rank=4, n_heads=8, n_kv=4:
      W_cQ [768,64] + A_q [16,4,48] + W_cK [768,32] + A_k [8,4,48]
      W_cV [768,32] + A_v [8,4,48] + W_O [384,768] + λ [8] + q_gain [8]
      = ~399K params (vs 2,359K for standard GQA at d=768)
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv: int,
        rank: int,
        rope_dims: int,
        rope_base: float,
        use_xsa: bool = False,
        num_iterations: int = 3,
    ):
        super().__init__()
        assert dim % n_heads == 0 and n_heads % n_kv == 0
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.rank = rank
        self.rope_dims = rope_dims
        self.use_xsa = use_xsa
        self.num_iterations = num_iterations
        self.d_head = dim // n_heads           # 96
        self.d_head_half = self.d_head // 2    # 48

        # Context projections (2D → Muon-eligible)
        self.W_cQ = CastedLinear(dim, 2 * n_heads * rank, bias=False)     # [768, 64]
        self.W_cK = CastedLinear(dim, 2 * n_kv * rank,   bias=False)     # [768, 32]
        self.W_cV = CastedLinear(dim, 2 * n_kv * rank,   bias=False)     # [768, 32]
        self.W_O  = CastedLinear(n_heads * self.d_head_half, dim, bias=False)  # [384, 768]

        # Basis tensors (3D → Adam; auto-passthrough in int6 quant, numel < 65536)
        self.A_q = nn.Parameter(torch.empty(2 * n_heads, rank, self.d_head_half))  # [16, 4, 48]
        self.A_k = nn.Parameter(torch.empty(2 * n_kv,   rank, self.d_head_half))  # [8, 4, 48]
        self.A_v = nn.Parameter(torch.empty(2 * n_kv,   rank, self.d_head_half))  # [8, 4, 48]

        # Differential lambda (1D → scalar Adam via ndim<2 rule)
        self.lam = nn.Parameter(torch.ones(n_heads))

        # Q-gain (matches "q_gain" control pattern → float32 scalar Adam)
        self.q_gain = nn.Parameter(torch.ones(n_heads, dtype=torch.float32))

        # RoPE cache (only for rope_dims dims of d_head_half)
        self.rotary = Rotary(rope_dims, base=rope_base)

        # XSA: additional output projection for last-iteration blocks
        if use_xsa:
            self.W_O_xsa = CastedLinear(n_heads * self.d_head_half, dim, bias=False)  # [384, 768]

    def forward(self, h: Tensor, x_residual: Tensor, layer_r: int) -> Tensor:
        """
        h           : [B, T, dim]  — pre-norm hidden state
        x_residual  : [B, T, dim]  — pre-norm residual stream (for XSA V)
        layer_r     : int          — current MoR iteration index
        """
        B, T, _ = h.shape

        # ---- Step 1: Factored Q/K/V reconstruction ----
        # W_cQ projects h to context codes, then einsum with static basis A_q
        c_Q = self.W_cQ(h).view(B, T, 2 * self.n_heads, self.rank)   # [B, T, 16, 4]
        # Broadcast einsum: code[b,t,head,r] × basis[head,r,d] → sum over r
        Q_all = (c_Q.unsqueeze(-1) * self.A_q).sum(-2)                # [B, T, 16, 48]
        Q1, Q2 = Q_all.chunk(2, dim=2)                                # each [B, T, 8, 48]

        c_K = self.W_cK(h).view(B, T, 2 * self.n_kv, self.rank)      # [B, T, 8, 4]
        K_all = (c_K.unsqueeze(-1) * self.A_k).sum(-2)                # [B, T, 8, 48]
        K1, K2 = K_all.chunk(2, dim=2)                                # each [B, T, 4, 48]

        c_V = self.W_cV(h).view(B, T, 2 * self.n_kv, self.rank)
        V_all = (c_V.unsqueeze(-1) * self.A_v).sum(-2)                # [B, T, 8, 48]
        V1, V2 = V_all.chunk(2, dim=2)                                # each [B, T, 4, 48]

        # ---- Step 2: Partial RoPE on reconstructed Q/K ----
        # Rotary is initialized with rope_dims; apply_rotary_emb handles partial rotation
        cos, sin = self.rotary(T, h.device, h.dtype)   # [1, T, 1, rope_dims//2]
        Q1 = apply_rotary_emb(Q1, cos, sin, self.rope_dims)
        Q2 = apply_rotary_emb(Q2, cos, sin, self.rope_dims)
        K1 = apply_rotary_emb(K1, cos, sin, self.rope_dims)
        K2 = apply_rotary_emb(K2, cos, sin, self.rope_dims)

        # ---- Step 3: QK-norm + Q-gain ----
        Q1 = F.rms_norm(Q1, (Q1.size(-1),))
        Q2 = F.rms_norm(Q2, (Q2.size(-1),))
        K1 = F.rms_norm(K1, (K1.size(-1),))
        K2 = F.rms_norm(K2, (K2.size(-1),))
        q_g = self.q_gain.to(dtype=Q1.dtype)[None, None, :, None]     # [1, 1, 8, 1]
        Q1 = Q1 * q_g
        Q2 = Q2 * q_g

        # ---- Step 4: XSA path (last iteration, XSA blocks only) ----
        if self.use_xsa and layer_r == self.num_iterations - 1:
            # XSA: use first d_head_half dims of residual stream as V values
            # x_residual [B, T, 768] → [B, T, 8, 48] by splitting the first 384 dims
            x_as_v = x_residual[..., :self.n_heads * self.d_head_half]
            x_as_v = x_as_v.view(B, T, self.n_heads, self.d_head_half)
            # Expand K1 [B, T, 4, 48] → [B, T, 8, 48] for full-head attention
            K1_exp = K1.repeat_interleave(self.n_heads // self.n_kv, dim=2)
            # flash_attn_3_func takes [B, T, H, D] format
            out = flash_attn_3_func(Q1, K1_exp, x_as_v, causal=True)  # [B, T, 8, 48]
            out = out.reshape(B, T, self.n_heads * self.d_head_half)   # [B, T, 384]
            return self.W_O_xsa(out)                                   # [B, T, 768]

        # ---- Step 5: Differential attention ----
        # FA3 handles GQA natively (Q:8 heads, K/V:4 heads → output:8 heads)
        out1 = flash_attn_3_func(Q1, K1, V1, causal=True)             # [B, T, 8, 48]
        out2 = flash_attn_3_func(Q2, K2, V2, causal=True)             # [B, T, 8, 48]
        lam  = self.lam.to(dtype=h.dtype)[None, None, :, None]        # [1, 1, 8, 1]
        out  = out1 - lam * out2                                       # [B, T, 8, 48]

        # ---- Step 6: GroupNorm + output projection ----
        # Reshape to [B*T, 384] for group_norm (C=384, num_groups=8 → 48 dims/group)
        out = F.group_norm(
            out.reshape(B * T, self.n_heads * self.d_head_half),
            num_groups=self.n_heads,
        ).reshape(B, T, self.n_heads * self.d_head_half)
        return self.W_O(out)                                           # [B, T, 768]
```

- [ ] **Step 1: Write failing test**

Create `test_dtpa.py`:

```python
"""Test DTPA shapes, param count, and basic differentiability."""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
# Patch flash_attn_interface to use F.scaled_dot_product_attention for CPU tests
import torch, torch.nn.functional as F

def _cpu_flash_attn(q, k, v, causal=False):
    # q,k,v in [B, T, H, D] format; convert to [B, H, T, D] for SDPA
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    # GQA: expand k,v to match q head count
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # back to [B, T, H, D]

import types
fa3_mod = types.SimpleNamespace(flash_attn_func=_cpu_flash_attn)
sys.modules['flash_attn_interface'] = fa3_mod

import train_gpt as tg

dim, n_heads, n_kv, rank, rope_dims = 64, 4, 2, 2, 8
B, T = 2, 16

dtpa = tg.DTPA(dim, n_heads, n_kv, rank, rope_dims, 10000.0, use_xsa=False, num_iterations=3)
h = torch.randn(B, T, dim, requires_grad=True)
x_res = torch.randn(B, T, dim)
out = dtpa(h, x_res, layer_r=0)
assert out.shape == (B, T, dim), f"Expected ({B},{T},{dim}), got {out.shape}"
out.sum().backward()
assert h.grad is not None, "No gradient on h"

# Test XSA path
dtpa_xsa = tg.DTPA(dim, n_heads, n_kv, rank, rope_dims, 10000.0, use_xsa=True, num_iterations=3)
out_xsa = dtpa_xsa(h.detach(), x_res, layer_r=2)  # last iteration triggers XSA
assert out_xsa.shape == (B, T, dim), f"XSA: Expected ({B},{T},{dim}), got {out_xsa.shape}"

# Param count sanity check (rough: should be much less than standard GQA)
n_dtpa = sum(p.numel() for p in dtpa.parameters())
n_std_gqa = dim*dim + dim*(dim//2)*2 + dim*dim  # Q + 2*(KV) + O at d=dim
assert n_dtpa < n_std_gqa, f"DTPA ({n_dtpa}) should be < standard GQA ({n_std_gqa})"
print(f"PASS: DTPA OK  params={n_dtpa}  std_gqa_params={n_std_gqa}")
```

- [ ] **Step 2: Run test to verify it FAILS**

```bash
python test_dtpa.py
```
Expected: `AttributeError: module 'train_gpt' has no attribute 'DTPA'`

- [ ] **Step 3: Implement `DTPA`** in `train_gpt.py` (after SwiGLU, replacing CausalSelfAttention)

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_dtpa.py
```
Expected: `PASS: DTPA OK  params=<N>  std_gqa_params=<M>` where N < M

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/
git commit -m "feat(helix): add DTPA (Differential Tensor Product Attention)"
```

---

## Task 5: Implement HELIXBlock

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py`
- Insert after `DTPA`, replacing SOTA `Block` class

**Full `HELIXBlock` implementation:**

```python
class HELIXBlock(nn.Module):
    """
    Single HELIX block. Shared across up to R iterations via MoR.

    Per-iteration adaptation: each of the R iterations gets its own
    attn_scale[r], mlp_scale[r], resid_mix[r] — matching CONTROL_TENSOR_NAME_PATTERNS.
    Peri-LN: pre-norm + post-norm (sandwich) at every sublayer.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv: int,
        rank: int,
        ffn_hidden: int,
        rope_dims: int,
        rope_base: float,
        use_xsa: bool,
        num_iterations: int,
        block_idx: int,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.block_idx = block_idx

        # D-TPA attention
        self.dtpa = DTPA(
            dim, n_heads, n_kv, rank, rope_dims, rope_base,
            use_xsa=use_xsa, num_iterations=num_iterations,
        )
        # SwiGLU FFN
        self.swiglu = SwiGLU(dim, ffn_hidden)

        # Peri-LN: 4 RMSNorm instances (2 per sublayer)
        self.pre_norm_attn  = RMSNorm()
        self.post_norm_attn = RMSNorm()
        self.pre_norm_mlp   = RMSNorm()
        self.post_norm_mlp  = RMSNorm()

        # Per-iteration adaptation scalars (named to match CONTROL_TENSOR_NAME_PATTERNS)
        # "attn_scale", "mlp_scale", "resid_mix" → float32, scalar AdamW
        self.iter_attn_scale = nn.ParameterList([
            nn.Parameter(torch.ones(dim, dtype=torch.float32)) for _ in range(num_iterations)
        ])
        self.iter_mlp_scale = nn.ParameterList([
            nn.Parameter(torch.ones(dim, dtype=torch.float32)) for _ in range(num_iterations)
        ])
        # resid_mix[r] is [2, dim]: mix[0]*x + mix[1]*x0 (anchored blending)
        self.iter_resid_mix = nn.ParameterList([
            nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
            for _ in range(num_iterations)
        ])

    def forward(self, x: Tensor, x0: Tensor, r: int) -> Tensor:
        """
        x  : [B, T, dim]  — current hidden state
        x0 : [B, T, dim]  — initial hidden state (iteration anchor for resid_mix)
        r  : int           — current MoR iteration index (0..num_iterations-1)
        """
        # Residual-mix: iteration-specific blend of current state and initial anchor
        mix = self.iter_resid_mix[r].to(dtype=x.dtype)       # [2, dim]
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention sublayer (Peri-LN: pre + post)
        h = self.pre_norm_attn(x)
        h = self.dtpa(h, x_residual=x, layer_r=r)
        h = self.post_norm_attn(h)
        attn_s = self.iter_attn_scale[r].to(dtype=x.dtype)[None, None, :]
        x = x + attn_s * h

        # MLP sublayer (Peri-LN: pre + post)
        h = self.pre_norm_mlp(x)
        h = self.swiglu(h)
        h = self.post_norm_mlp(h)
        mlp_s = self.iter_mlp_scale[r].to(dtype=x.dtype)[None, None, :]
        x = x + mlp_s * h

        return x
```

- [ ] **Step 1: Write failing test**

Create `test_helixblock.py`:

```python
"""Test HELIXBlock forward and backward."""
import sys, os, types, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))

def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)
sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)

import train_gpt as tg

dim, n_heads, n_kv, rank, rope_dims = 64, 4, 2, 2, 8
B, T, R = 2, 16, 3

block = tg.HELIXBlock(dim, n_heads, n_kv, rank, 128, rope_dims, 10000.0,
                       use_xsa=False, num_iterations=R, block_idx=0)
x  = torch.randn(B, T, dim, requires_grad=True)
x0 = torch.randn(B, T, dim)

for r in range(R):
    out = block(x, x0, r)
    assert out.shape == (B, T, dim), f"r={r}: expected ({B},{T},{dim}), got {out.shape}"

out.sum().backward()
assert x.grad is not None, "No gradient on x"

# Check per-iteration params exist
assert len(block.iter_attn_scale) == R
assert len(block.iter_mlp_scale) == R
assert len(block.iter_resid_mix) == R
print(f"PASS: HELIXBlock OK  n_iters={R}")
```

- [ ] **Step 2: Run test to verify it FAILS**

```bash
python test_helixblock.py
```
Expected: `AttributeError: module 'train_gpt' has no attribute 'HELIXBlock'`

- [ ] **Step 3: Implement `HELIXBlock`** in `train_gpt.py` (after DTPA, replacing SOTA `Block`)

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_helixblock.py
```
Expected: `PASS: HELIXBlock OK  n_iters=3`

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/
git commit -m "feat(helix): add HELIXBlock with Peri-LN and per-iteration adaptation"
```

---

## Task 6: Implement HELIX_GPT

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py`
- Insert after `HELIXBlock`, replacing SOTA `GPT` class

**Full `HELIX_GPT` implementation:**

```python
class HELIX_GPT(nn.Module):
    """
    HELIX: 5 unique blocks × 3 MoR iterations = 15 virtual layers.
    d=768, D-TPA attention, SwiGLU FFN, Peri-LN, U-Net skips, MoR aux loss.
    Interface contract (for SOTA training loop):
      base_model.blocks          — nn.ModuleList of HELIXBlock
      base_model.smear           — SmearGate
      base_model.bigram          — BigramHashEmbedding
      base_model.tok_emb         — nn.Embedding (tied with lm_head.T)
      base_model.skip_weights    — nn.Parameter
      base_model.mtp_heads       — nn.ModuleList([])
      base_model.mtp_num_heads   — 0
      base_model.forward_logits  — [B,T,V] for eval
      base_model.lm_head         — None (tied embeddings)
      base_model.mor_gate        — nn.ParameterList (2 gates)
    """
    def __init__(
        self,
        vocab_size: int,
        num_unique_blocks: int,
        num_iterations: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dtpa_rank: int,
        ffn_hidden: int,
        rope_dims: int,
        xsa_last_n: int,
        bigram_vocab_size: int,
        bigram_dim: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.num_unique_blocks = num_unique_blocks
        self.num_iterations    = num_iterations
        self.model_dim         = model_dim
        self.vocab_size        = vocab_size
        self.logit_softcap     = logit_softcap
        self.num_skip          = 2  # U-Net: collect/inject first 2 blocks

        # Token embedding (tied with output projection)
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # BigramHash embedding for 2-gram context signal
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)

        # SmearGate: causal 1-token blend after embedding
        self.smear = SmearGate(model_dim)

        # Input RMSNorm (applied before smear gate)
        self.input_norm = RMSNorm()

        # 5 unique HELIXBlocks
        self.blocks = nn.ModuleList([
            HELIXBlock(
                dim=model_dim,
                n_heads=num_heads,
                n_kv=num_kv_heads,
                rank=dtpa_rank,
                ffn_hidden=ffn_hidden,
                rope_dims=rope_dims,
                rope_base=rope_base,
                use_xsa=(i >= num_unique_blocks - xsa_last_n),
                num_iterations=num_iterations,
                block_idx=i,
            )
            for i in range(num_unique_blocks)
        ])

        # U-Net skip weights (one per skip connection = 2 total)
        # Named "skip_weights" → matches CONTROL_TENSOR_NAME_PATTERNS
        self.skip_weights = nn.Parameter(torch.zeros(self.num_skip, model_dim, dtype=torch.float32))

        # Final layer norm
        self.final_norm = RMSNorm()

        # Tied output projection (None when tied; lm_head used by training loop check)
        self.lm_head = None
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std

        # MoR inter-iteration gate vectors (R-1 = 2 gates)
        # Named "mor_gate.N" → "mor_gate" in CONTROL_TENSOR_NAME_PATTERNS → scalar AdamW
        self.mor_gate = nn.ParameterList([
            nn.Parameter(torch.zeros(model_dim, 1))
            for _ in range(num_iterations - 1)
        ])
        self._current_lb_weight: float = 0.0  # set by training loop

        # MTP interface stubs (unused, required by training loop)
        self.mtp_heads = nn.ModuleList([])
        self.mtp_num_heads = 0

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init + muP output scaling. Lambda init per block."""
        virtual_depth = self.num_unique_blocks * self.num_iterations  # 15
        output_scale  = 1.0 / math.sqrt(2 * virtual_depth)           # 1/sqrt(30)

        # CastedLinear weights: orthogonal + muP scaling (zero-init for output projections)
        for module in self.modules():
            if isinstance(module, CastedLinear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight)
                    module.weight.data.mul_(output_scale)

        # TPA basis tensors: orthogonal per head slice
        for name, param in self.named_parameters():
            if any(x in name for x in ('A_q', 'A_k', 'A_v')):
                n_slices = param.shape[0]
                for i in range(n_slices):
                    nn.init.orthogonal_(param.data[i])  # [rank, d_head_half]

        # Lambda (differential): per-block depth-aware init
        for block_idx, block in enumerate(self.blocks):
            lam_val = 0.8 - 0.6 * math.exp(-0.3 * block_idx)
            block.dtpa.lam.data.fill_(lam_val)

        # Token embedding
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, 0.0, self.tied_embed_init_std)

    def _embed(self, input_ids: Tensor) -> Tensor:
        """Embed token IDs: tok_emb + bigram + input_norm + smear."""
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = self.input_norm(x)
        x = self.smear(x)
        return x

    def _project_logits(self, x: Tensor) -> Tensor:
        """Tied output projection + softcap."""
        logits = F.linear(x, self.tok_emb.weight.to(x.dtype))
        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits

    def _mor_aux_loss(self, gate_logits: list[Tensor]) -> Tensor:
        """Load-balancing loss: push each gate toward routing 1/3 of tokens out."""
        p0 = torch.sigmoid(gate_logits[0]).mean()
        p1 = torch.sigmoid(gate_logits[1]).mean()
        target = 1.0 / 3.0
        w = self._current_lb_weight
        return w * ((p0 - target) ** 2 + (p1 - target) ** 2)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Training forward. Returns ce_loss + mor_aux_loss."""
        x  = self._embed(input_ids)   # [B, T, d]
        x0 = x.clone()                # anchor for resid_mix
        first_iter_hidden: list[Tensor] = []
        mor_gate_logits:   list[Tensor] = []

        for r in range(self.num_iterations):
            for k, block in enumerate(self.blocks):
                # U-Net: inject encoder skips at last iteration
                if r == self.num_iterations - 1 and k < self.num_skip:
                    skip_idx = self.num_skip - 1 - k
                    w = self.skip_weights[skip_idx].to(dtype=x.dtype)
                    x = x + w[None, None, :] * first_iter_hidden[skip_idx]

                x = block(x, x0, r)

                # U-Net: collect first-iteration hiddens (no detach — full BPTT)
                if r == 0 and k < self.num_skip:
                    first_iter_hidden.append(x)

            # MoR gate between iterations (not at last iteration)
            if r < self.num_iterations - 1:
                gate_logit = x @ self.mor_gate[r].to(dtype=x.dtype)   # [B, T, 1]
                mor_gate_logits.append(gate_logit)

        x = self.final_norm(x)

        # Cross-entropy loss
        logits = self._project_logits(x)
        ce_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size).float(),
            target_ids.view(-1),
        )

        # MoR auxiliary load-balancing loss
        aux_loss = self._mor_aux_loss(mor_gate_logits)
        return ce_loss + aux_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """
        Inference-only forward: all 3 iterations, MoR gates inactive.
        Used by sliding-window eval (eval_val_sliding).
        Returns [B, T, V] logits.
        """
        x  = self._embed(input_ids)
        x0 = x.clone()
        first_iter_hidden: list[Tensor] = []

        for r in range(self.num_iterations):
            for k, block in enumerate(self.blocks):
                if r == self.num_iterations - 1 and k < self.num_skip:
                    skip_idx = self.num_skip - 1 - k
                    w = self.skip_weights[skip_idx].to(dtype=x.dtype)
                    x = x + w[None, None, :] * first_iter_hidden[skip_idx]
                x = block(x, x0, r)
                if r == 0 and k < self.num_skip:
                    first_iter_hidden.append(x)

        x = self.final_norm(x)
        return self._project_logits(x)
```

- [ ] **Step 1: Write failing test**

Create `test_helix_gpt.py`:

```python
"""Test HELIX_GPT forward, backward, interface contract, and param count."""
import sys, os, types, math
sys.path.insert(0, os.path.dirname(__file__))

import torch, torch.nn.functional as F

def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)
sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)

import train_gpt as tg

V, K, R, d = 1024, 5, 3, 64
n_heads, n_kv, rank = 4, 2, 2
ffn_h, rope_d = 128, 8
B, T = 2, 32

model = tg.HELIX_GPT(
    vocab_size=V, num_unique_blocks=K, num_iterations=R,
    model_dim=d, num_heads=n_heads, num_kv_heads=n_kv,
    dtpa_rank=rank, ffn_hidden=ffn_h, rope_dims=rope_d,
    xsa_last_n=2, bigram_vocab_size=512, bigram_dim=32,
    tie_embeddings=True, tied_embed_init_std=0.005,
    logit_softcap=30.0,
)

ids = torch.randint(0, V, (B, T))
tgt = torch.randint(0, V, (B, T))

# Training forward
model._current_lb_weight = 0.01
loss = model(ids, tgt)
assert loss.shape == (), f"loss should be scalar, got {loss.shape}"
loss.backward()

# Inference forward
model.eval()
with torch.no_grad():
    logits = model.forward_logits(ids)
assert logits.shape == (B, T, V), f"Expected ({B},{T},{V}), got {logits.shape}"

# Interface contract
assert hasattr(model, 'blocks') and len(model.blocks) == K
assert hasattr(model, 'smear')
assert hasattr(model, 'bigram')
assert hasattr(model, 'skip_weights')
assert hasattr(model, 'mtp_heads') and len(model.mtp_heads) == 0
assert model.mtp_num_heads == 0
assert model.lm_head is None
assert len(model.mor_gate) == R - 1

n_params = sum(p.numel() for p in model.parameters())
print(f"PASS: HELIX_GPT OK  params={n_params:,}")
```

- [ ] **Step 2: Run test to verify it FAILS**

```bash
python test_helix_gpt.py
```
Expected: `AttributeError: module 'train_gpt' has no attribute 'HELIX_GPT'`

- [ ] **Step 3: Implement `HELIX_GPT`** in `train_gpt.py` (replacing SOTA `GPT` class)

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_helix_gpt.py
```
Expected: `PASS: HELIX_GPT OK  params=<N>` (N should be around 20M when d=768 with full config)

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/
git commit -m "feat(helix): add HELIX_GPT with MoR, U-Net skips, aux loss, forward_logits"
```

---

## Task 7: Update main() — Model Instantiation, Optimizer Routing, and compile Flag

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py` (main() function)

**Changes required:**

### 7.1 Replace GPT instantiation

Find the `base_model = GPT(...)` call and replace with:

```python
base_model = HELIX_GPT(
    vocab_size=args.vocab_size,
    num_unique_blocks=args.num_unique_blocks,
    num_iterations=args.num_iterations,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    dtpa_rank=args.dtpa_rank,
    ffn_hidden=args.ffn_hidden,
    rope_dims=args.rope_dims,
    xsa_last_n=args.xsa_last_n,
    bigram_vocab_size=args.bigram_vocab_size,
    bigram_dim=args.bigram_dim,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
).to(device).bfloat16()
# Keep CastedLinear weights in float32 (cast at compute time)
for module in base_model.modules():
    if isinstance(module, CastedLinear):
        module.float()
restore_low_dim_params_to_fp32(base_model)
```

Remove these SOTA-specific lines that reference bank tensors (they don't exist in HELIX):
```python
# DELETE these lines:
base_model.qo_bank.data = base_model.qo_bank.data.float()
base_model.kv_bank.data = base_model.kv_bank.data.float()
base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
```

### 7.2 Change fullgraph=True → fullgraph=False

```python
# Find:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
# Change to:
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
```

### 7.3 Update optimizer routing

Replace the SOTA optimizer split (which references `qo_bank`, `kv_bank` etc.) with:

```python
# --- HELIX Optimizer routing ---
# 2D weight matrices from blocks (DTPA projections + SwiGLU weights) → Parallel Muon
# Excludes: control-pattern tensors, 3D basis tensors (A_q/A_k/A_v)
all_block_named = list(base_model.blocks.named_parameters())
matrix_params = [
    p for n, p in all_block_named
    if p.ndim == 2
    and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    and not any(x in n for x in ('A_q', 'A_k', 'A_v'))
]

# Scalar/control params from blocks (norms, scales, mix, lambda, q_gain) → scalar AdamW
scalar_params = [
    p for n, p in all_block_named
    if p.ndim != 2
    or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    or any(x in n for x in ('A_q', 'A_k', 'A_v'))
]

# MoR gate params (on base_model, not in blocks) → scalar AdamW
# These are named "mor_gate.0", "mor_gate.1" → "mor_gate" in CONTROL_PATTERNS
for p in base_model.mor_gate.parameters():
    scalar_params.append(p)

# Model-level scalars
scalar_params.append(base_model.smear.gate)
scalar_params.append(base_model.bigram.scale)
if base_model.bigram.proj is not None:
    scalar_params.append(base_model.bigram.proj.weight)
scalar_params.append(base_model.skip_weights)

token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
tok_params = [
    {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr},
    {"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr},
]

# replicated_params: manually all-reduced before Adam steps (everything not in Muon)
replicated_params = [p for pg in tok_params for p in pg["params"]]
replicated_params.extend(scalar_params)

optimizer_tok = torch.optim.AdamW(
    tok_params,
    betas=(args.beta1, args.beta2),
    eps=args.adam_eps,
    weight_decay=args.adam_wd,
    fused=True,
)
optimizer_muon = Muon(
    matrix_params,
    lr=args.matrix_lr,
    momentum=args.muon_momentum,
    backend_steps=args.muon_backend_steps,
    weight_decay=args.muon_wd,
)
for group in optimizer_muon.param_groups:
    group["base_lr"] = args.matrix_lr

optimizer_scalar = torch.optim.AdamW(
    [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
    betas=(args.beta1, args.beta2),
    eps=args.adam_eps,
    weight_decay=args.adam_wd,
    fused=True,
)
optimizer_head = None  # HELIX uses tied embeddings
optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
```

### 7.4 Add MoR lb_weight decay in training loop

In the training loop, after computing `scale` (the LR multiplier), add:

```python
# MoR load-balancing weight: starts at mor_lb_weight, decays to 0 during warmdown
# Only active when scale < 1.0 (warmdown phase)
if args.warmdown_iters > 0 and scale < 1.0:
    # How far into warmdown: 0.0 = warmdown start, 1.0 = end of training
    warmdown_frac = 1.0 - scale  # scale goes 1→0, frac goes 0→1
    # Decay over last mor_lb_decay_steps steps
    decay_frac = min(1.0, warmdown_frac * args.warmdown_iters / max(args.mor_lb_decay_steps, 1))
    base_model._current_lb_weight = args.mor_lb_weight * (1.0 - decay_frac)
else:
    base_model._current_lb_weight = args.mor_lb_weight
```

Also update the MTP-related log line that references `base_model.mtp_heads`:
```python
# Find: log0(f"mtp_num_heads:{args.mtp_num_heads} ...")
# The HELIX model has mtp_num_heads=0, so this line is fine as-is
```

### 7.5 Update EMA state dict logging

The SOTA logs `xsa_layers` like:
```python
xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
```
HELIX uses `b.dtpa.use_xsa`, not `b.attn.use_xsa`. Update:
```python
xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.dtpa.use_xsa]
```

- [ ] **Step 1: Write failing test**

Create `test_optimizer_routing.py`:

```python
"""Verify optimizer routing: correct params in each optimizer group."""
import sys, os, types, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)
import train_gpt as tg

V, K, R, d = 1024, 5, 3, 64
model = tg.HELIX_GPT(
    vocab_size=V, num_unique_blocks=K, num_iterations=R,
    model_dim=d, num_heads=4, num_kv_heads=2, dtpa_rank=2,
    ffn_hidden=128, rope_dims=8, xsa_last_n=2,
    bigram_vocab_size=512, bigram_dim=32,
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
)

all_block_named = list(model.blocks.named_parameters())
matrix_params = [
    p for n, p in all_block_named
    if p.ndim == 2
    and not any(pat in n for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
    and not any(x in n for x in ('A_q', 'A_k', 'A_v'))
]
scalar_params = [
    p for n, p in all_block_named
    if p.ndim != 2
    or any(pat in n for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
    or any(x in n for x in ('A_q', 'A_k', 'A_v'))
]
for p in model.mor_gate.parameters():
    scalar_params.append(p)

# All block params should be in exactly one group
all_block_params = list(model.blocks.parameters())
routed = set(id(p) for p in matrix_params) | set(id(p) for p in scalar_params)
unrouted = [p for p in all_block_params if id(p) not in routed]
assert len(unrouted) == 0, f"{len(unrouted)} block params not routed to any optimizer"

# matrix_params should all be 2D
for p in matrix_params:
    assert p.ndim == 2, f"Non-2D in matrix_params: shape={p.shape}"

# No A_q/A_k/A_v in matrix_params
for n, p in all_block_named:
    if any(x in n for x in ('A_q', 'A_k', 'A_v')):
        assert not any(p is mp for mp in matrix_params), f"{n} must not be in Muon"

print(f"PASS: Optimizer routing OK  muon={len(matrix_params)} scalar={len(scalar_params)}")
```

- [ ] **Step 2: Run test to verify it FAILS** (references old GPT/bank setup)

```bash
python test_optimizer_routing.py
```
Expected: Various errors since main() still has old GPT code

- [ ] **Step 3: Apply all main() changes** as described in 7.1–7.5

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_optimizer_routing.py
```
Expected: `PASS: Optimizer routing OK  muon=<N> scalar=<M>`

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
git commit -m "feat(helix): update main() with HELIX_GPT, Parallel Muon routing, MoR lb decay"
```

---

## Task 8: Update Serialization (_classify_param and eval_model)

**Files:**
- Modify: `records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py` (lines ~1234–1380 and ~1780+)

### 8.1 Update `_classify_param` for HELIX naming

```python
# Find the existing _classify_param function and replace with:
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    # HELIX SwiGLU FFN + SOTA MLP
    if ".swiglu." in name or ".mlp." in name:
        return "mlp"
    # HELIX D-TPA + SOTA attention (W_O is >65536 numel → int6)
    if ".dtpa." in name or ".attn." in name or (
        ".proj." in name
        and ".mlp." not in name
        and ".swiglu." not in name
    ):
        return "attn"
    return "other"
```

This ensures:
- `blocks.N.dtpa.W_O.weight` [384×768 = 294K params] → "attn" → int6 ✓
- `blocks.N.swiglu.gate.weight` [768×1536 = 1.18M params] → "mlp" → int6 ✓
- `blocks.N.swiglu.fc.weight` → "mlp" → int6 ✓
- `blocks.N.swiglu.proj.weight` → "mlp" → int6 ✓
- W_cQ/W_cK/W_cV (< 65536 numel) → auto-passthrough as fp16 regardless ✓
- A_q/A_k/A_v (< 65536 numel) → auto-passthrough as fp16 regardless ✓

### 8.2 Update artifact serialization in main()

**Compression note:** CLAUDE.md states "int6+zstd-22" but the actual SOTA code uses `lzma.compress(quant_raw, preset=6)` (visible at SOTA line 1795 and log message "int6+lzma"). The SOTA submission achieves ~0.726 bytes/param with lzma. HELIX must use the same compression to achieve the same ratio and keep the artifact under 16MB. Do NOT switch to zstd.

Find the serialization block that calls `_unbank_state_dict` and `_rebank_state_dict`. For HELIX, these functions are not needed (no bank tensors). Replace:

```python
# FIND (approx. lines 1788–1835):
# sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
# unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
# quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
# ...
# deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
# deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

# REPLACE WITH:
sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
# HELIX has no 3D bank tensors — quantize the state dict directly
quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
quant_buf = io.BytesIO()
torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
quant_raw = quant_buf.getvalue()
quant_blob = lzma.compress(quant_raw, preset=6)
if master_process:
    with open("final_model.int6.ptz", "wb") as f:
        f.write(quant_blob)
    ...
quant_state = torch.load(
    io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu",
)
deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
```

### 8.3 Update eval_model instantiation

Find the `eval_model = GPT(...)` call and replace with:

```python
eval_model = HELIX_GPT(
    vocab_size=args.vocab_size,
    num_unique_blocks=args.num_unique_blocks,
    num_iterations=args.num_iterations,
    model_dim=args.model_dim,
    num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads,
    dtpa_rank=args.dtpa_rank,
    ffn_hidden=args.ffn_hidden,
    rope_dims=args.rope_dims,
    xsa_last_n=args.xsa_last_n,
    bigram_vocab_size=args.bigram_vocab_size,
    bigram_dim=args.bigram_dim,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
).to(device).bfloat16()
for m in eval_model.modules():
    if isinstance(m, CastedLinear):
        m.float()
restore_low_dim_params_to_fp32(eval_model)
eval_model.load_state_dict(deq_state, strict=True)
# Eval model CAN use fullgraph=True (no Python loops that escape the graph)
# However, MoR forward still has Python loops → keep fullgraph=False for safety
compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=False)
```

- [ ] **Step 1: Write failing test**

Create `test_serialize.py`:

```python
"""Test that _classify_param correctly classifies HELIX param names."""
import sys, os, types
sys.path.insert(0, os.path.dirname(__file__))
sys.modules.setdefault('flash_attn_interface', types.SimpleNamespace(flash_attn_func=None))
import train_gpt as tg

cases = [
    ("blocks.0.dtpa.W_O.weight",         "attn"),
    ("blocks.0.swiglu.gate.weight",       "mlp"),
    ("blocks.0.swiglu.fc.weight",         "mlp"),
    ("blocks.0.swiglu.proj.weight",       "mlp"),
    ("tok_emb.weight",                    "embed"),
    ("bigram.embed.weight",               "embed" if "tok_emb" in "bigram.embed.weight" else "other"),
    ("blocks.0.pre_norm_attn.weight",     "other"),
]
# Fix: bigram.embed is not tok_emb/lm_head
cases[5] = ("bigram.embed.weight", "other")

for name, expected in cases:
    got = tg._classify_param(name)
    assert got == expected, f"_classify_param({name!r}) = {got!r}, expected {expected!r}"
print("PASS: _classify_param OK")
```

- [ ] **Step 2: Run test to verify it FAILS**

```bash
python test_serialize.py
```
Expected: `AssertionError: _classify_param('blocks.0.dtpa.W_O.weight') = 'other', expected 'attn'`

- [ ] **Step 3: Apply serialization changes** (8.1, 8.2, 8.3)

- [ ] **Step 4: Run test to verify it PASSES**

```bash
python test_serialize.py
```
Expected: `PASS: _classify_param OK`

- [ ] **Step 5: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
git commit -m "feat(helix): update _classify_param and serialization for HELIX naming"
```

---

## Task 9: End-to-End CPU Smoke Test

**Files:**
- Create: `records/track_10min_16mb/2026-03-25_HELIX/test_smoke.py`

This test verifies the full training forward/backward cycle on CPU with tiny model sizes, ensuring no crashes before committing to a full GPU run.

```python
"""
Full forward/backward/optimizer smoke test on CPU.
Uses tiny model (d=64) to verify the entire pipeline without GPUs.
"""
import sys, os, types, math, io
sys.path.insert(0, os.path.dirname(__file__))
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

import torch, torch.nn.functional as F

def _cpu_fa3(q, k, v, causal=False):
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    if k.size(1) != q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
sys.modules['flash_attn_interface'] = types.SimpleNamespace(flash_attn_func=_cpu_fa3)

import train_gpt as tg

V, K, R, d = 1024, 3, 2, 64
B, T = 2, 32

model = tg.HELIX_GPT(
    vocab_size=V, num_unique_blocks=K, num_iterations=R,
    model_dim=d, num_heads=4, num_kv_heads=2, dtpa_rank=2,
    ffn_hidden=128, rope_dims=8, xsa_last_n=1,
    bigram_vocab_size=256, bigram_dim=32,
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
)

# --- Training step ---
model.train()
model._current_lb_weight = 0.01
ids  = torch.randint(0, V, (B, T))
tgt  = torch.randint(0, V, (B, T))

loss = model(ids, tgt)
assert torch.isfinite(loss), f"Loss not finite: {loss.item()}"
print(f"  forward OK  loss={loss.item():.4f}")

loss.backward()
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        print(f"  WARN: no grad for {name}")
print("  backward OK")

# --- Optimizer step (verifies param routing) ---
all_block_named = list(model.blocks.named_parameters())
matrix_params = [
    p for n, p in all_block_named
    if p.ndim == 2
    and not any(pat in n for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
    and not any(x in n for x in ('A_q', 'A_k', 'A_v'))
]
scalar_params = [
    p for n, p in all_block_named
    if p.ndim != 2
    or any(pat in n for pat in tg.CONTROL_TENSOR_NAME_PATTERNS)
    or any(x in n for x in ('A_q', 'A_k', 'A_v'))
]
for p in model.mor_gate.parameters():
    scalar_params.append(p)
scalar_params.extend([model.smear.gate, model.bigram.scale, model.skip_weights])
if model.bigram.proj is not None:
    scalar_params.append(model.bigram.proj.weight)

opt_adam = torch.optim.AdamW(scalar_params + [model.tok_emb.weight, model.bigram.embed.weight], lr=1e-3)
opt_adam.step()
opt_adam.zero_grad()
print("  optimizer step OK")

# --- Inference forward ---
model.eval()
with torch.no_grad():
    logits = model.forward_logits(ids)
assert logits.shape == (B, T, V)
print(f"  inference OK  logits.shape={logits.shape}")

# --- Quantization dry-run (no CUDA needed) ---
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
quant_result, quant_meta = tg.mixed_quantize_int6(sd, {"mlp", "attn"})
deq_sd = tg.dequantize_mixed_int6(quant_result, quant_meta, sd)
# Verify deq_sd has same keys as sd
assert set(deq_sd.keys()) == set(sd.keys()), "Key mismatch after quantization roundtrip"
print("  quantization roundtrip OK")

n_params = sum(p.numel() for p in model.parameters())
print(f"\nPASS: Smoke test OK  params={n_params:,}  loss={loss.item():.4f}")
```

- [ ] **Step 1: Run smoke test**

```bash
cd records/track_10min_16mb/2026-03-25_HELIX
python test_smoke.py
```
Expected output:
```
  forward OK  loss=<value>
  backward OK
  optimizer step OK
  inference OK  logits.shape=torch.Size([2, 32, 1024])
  quantization roundtrip OK

PASS: Smoke test OK  params=<N>  loss=<value>
```

- [ ] **Step 2: Fix any issues found** (shapes, missing attributes, quantization errors)

- [ ] **Step 3: Run full test suite**

```bash
python test_hparams.py && python test_swiglu.py && python test_dtpa.py \
  && python test_helixblock.py && python test_helix_gpt.py \
  && python test_optimizer_routing.py && python test_serialize.py \
  && python test_smoke.py
```
Expected: All tests print `PASS`

- [ ] **Step 4: Commit**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/
git commit -m "test(helix): add full smoke test suite; all tests passing on CPU"
```

---

## Task 10: Submission Files and Pre-Flight Check

**Files:**
- Create: `records/track_10min_16mb/2026-03-25_HELIX/submission.json`
- Create: `records/track_10min_16mb/2026-03-25_HELIX/README.md`

### 10.1 Pre-flight check: verify param count and artifact size

Add this snippet to `test_smoke.py` or run inline:

```python
# Full-size model param count estimate
model_full = tg.HELIX_GPT(
    vocab_size=1024, num_unique_blocks=5, num_iterations=3,
    model_dim=768, num_heads=8, num_kv_heads=4, dtpa_rank=4,
    ffn_hidden=1536, rope_dims=16, xsa_last_n=2,
    bigram_vocab_size=2048, bigram_dim=128,
    tie_embeddings=True, tied_embed_init_std=0.005, logit_softcap=30.0,
)
n = sum(p.numel() for p in model_full.parameters())
bytes_estimate = n * 0.726  # int6+zstd-22 compression ratio bytes/param
print(f"Full model: {n/1e6:.2f}M params  ~{bytes_estimate/1e6:.2f}MB artifact")
assert bytes_estimate < 16 * 1024 * 1024, f"Artifact too large: {bytes_estimate/1e6:.2f}MB"
print("PASS: artifact size estimate OK")
```

Run: `python -c "exec(open('test_smoke.py').read())"` after appending the snippet.

Expected: `Full model: ~20.91M params  ~15.18MB artifact` (must be < 16MB)

### 10.2 Create placeholder submission files

Create `submission.json` (to be filled after training completes):
```json
{
  "name": "HELIX: D-TPA + MoR + Peri-LN",
  "val_bpb": null,
  "bytes_total": null,
  "blurb": "HELIX architecture: 5 unique HELIXBlocks × 3 MoR iterations (15 virtual layers), d=768. D-TPA (Differential Tensor Product Attention, 399K/block vs 2359K GQA), SwiGLU(hidden=1536), Peri-LN (sandwich norm), U-Net skip connections, MoR load-balancing aux loss. Projections: LeakyReLU replaced by SwiGLU; attention: rank-4 factored QKV with differential noise cancellation. int6+lzma artifact.",
  "author": "helix-team",
  "github_id": "helix-team",
  "date": "2026-03-25"
}
```

### 10.3 Verify line count of train_gpt.py

```bash
wc -l records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
```

**Line count rule clarification:** CLAUDE.md says "Training script must stay under 1500 lines" but this refers to the top-level baseline `train_gpt.py` (the "Entry Points" table explicitly lists "≤1500 lines hard limit" for `train_gpt.py`). Competitive submissions in `records/` are NOT subject to this limit — the accepted SOTA submission at `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` is 1898 lines. Aim for ≤ 2100 lines for readability.

```bash
# Enforce generous upper bound (ensures we haven't accidentally duplicated code)
LINE_COUNT=$(wc -l < records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py)
[ "$LINE_COUNT" -le 2100 ] && echo "PASS: ${LINE_COUNT} lines ≤ 2100" || echo "WARN: ${LINE_COUNT} lines exceeds 2100, consider trimming"
```

### 10.4 Full GPU run command

When ready to run the full challenge on 8×H100:

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
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-25_HELIX/train_gpt.py
```

### 10.5 After training: fill submission.json

After training completes, update `submission.json` with:
- `val_bpb`: final sliding-window BPB from training log (`final_int6_roundtrip val_bpb:...`)
- `bytes_total`: artifact size (sum of `final_model.int6.ptz` + `train_gpt.py` code bytes)

- [ ] **Step 1: Run pre-flight param count check**

```bash
cd records/track_10min_16mb/2026-03-25_HELIX
python -c "
import sys, types, torch, torch.nn.functional as F
def _fa3(q,k,v,causal=False):
    q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
    if k.size(1)!=q.size(1):
        rep=q.size(1)//k.size(1);k=k.repeat_interleave(rep,1);v=v.repeat_interleave(rep,1)
    return F.scaled_dot_product_attention(q,k,v,is_causal=causal).transpose(1,2)
sys.modules['flash_attn_interface']=types.SimpleNamespace(flash_attn_func=_fa3)
import train_gpt as tg
m=tg.HELIX_GPT(1024,5,3,768,8,4,4,1536,16,2,2048,128,True,0.005,30.0)
n=sum(p.numel() for p in m.parameters())
est=n*0.726
print(f'params={n/1e6:.2f}M  artifact~={est/1e6:.2f}MB')
assert est < 16*1024*1024, f'TOO BIG: {est/1e6:.2f}MB'
print('OK')
"
```
Expected: `params=~20.91M  artifact~=15.18MB  OK`

- [ ] **Step 2: Create submission files**

- [ ] **Step 3: Commit all**

```bash
cd ../../..
git add records/track_10min_16mb/2026-03-25_HELIX/
git commit -m "feat(helix): submission files and pre-flight checks passing"
```

---

## Implementation Notes

### Key invariants to maintain

1. **No DDP wrapper** — Parallel Muon handles matrix grad sync via reduce-scatter/all-gather. Non-matrix params use manual `dist.all_reduce` via `replicated_params`.

2. **fullgraph=False always** — MoR's Python `for r, for k` loops and `first_iter_hidden` list prevent fullgraph compilation. This applies to both training and eval model.

3. **A_q/A_k/A_v are 3D** — They're excluded from Muon (needs 2D), go to scalar Adam. They're < 65K params each so auto-passthrough as fp16 in int6 quantization — no special reshape needed.

4. **mor_gate must be in CONTROL_TENSOR_NAME_PATTERNS** — Otherwise its 2D shape [768,1] would route it to Muon, breaking its gradient dynamics.

5. **U-Net skips: no detach** — `first_iter_hidden` tensors flow gradients through all iterations (BPTT). Detaching would block gradient flow from late iterations into early blocks.

6. **_current_lb_weight attribute** — Set by the training loop before each forward call. The `_mor_aux_loss` method reads this instance attribute at call time.

7. **XSA only at r=num_iterations-1** — XSA blocks (the last `xsa_last_n` blocks) gate on `layer_r == self.num_iterations - 1`. For all other iterations they run standard D-TPA.

8. **apply_rotary_emb expects [B, T, H, D]** — The SOTA convention (not [B, H, T, D] like the baseline). The Rotary cache returns `[1, T, 1, rope_dims//2]` which broadcasts correctly over B and H dims.

### Debugging tips

- If loss is NaN on first step: check orthogonal init scale (output_scale = 1/sqrt(30)); try reducing `tied_embed_init_std` to 0.001
- If artifact too large: check that W_cQ/W_cK/W_cV are classified as passthrough (< 65536 numel) and not double-counted in byte estimate
- If `dist.all_reduce` hangs: ensure `replicated_params` includes all non-Muon params (especially `mor_gate` parameters and A_q/A_k/A_v basis tensors)
- If `torch.compile` fails: remember `fullgraph=False` is required; if still failing check for unsupported ops in DTPA (F.group_norm should be fine)
