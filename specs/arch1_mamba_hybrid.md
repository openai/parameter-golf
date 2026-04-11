# Architecture 1: Hybrid Mamba-Attention — Tech Spec

## 1. Executive Summary

Replace most attention layers with Mamba-2 selective state-space layers to fit **18-20 total layers** in the same 16MB budget that currently holds 11 transformer layers. The key insight: attention matrices (Q/K/V/O projections) consume ~51% of the parameter budget. Mamba layers use ~3.3x fewer parameters per layer via state-space recurrence with no KV cache.

**Target BPB:** 1.095-1.115 (vs SOTA 1.1147)

---

## 2. Baseline Reference (SOTA)

| Component | SOTA Value | Source Location |
|-----------|-----------|-----------------|
| File | `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py` | 2135 lines |
| Layers | 11 (5 encoder + 6 decoder) | `train_gpt.py:47` `num_layers=11` |
| Model dim | 512 | `train_gpt.py:49` `model_dim=512` |
| Heads | 8Q / 4KV (GQA) | `train_gpt.py:48,50` |
| MLP mult | 3.0 (1536 hidden) | `train_gpt.py:51` |
| Vocab | 1024 (tied embeddings) | `train_gpt.py:46` |
| Parameter banks | `qo_bank[22,512,512]`, `kv_bank[22,256,512]`, `mlp_up_bank[11,1536,512]`, `mlp_down_bank[11,512,1536]` | `train_gpt.py:831-834` |
| Total params | ~27M | submission.json |
| Quantization | Int6 GPTQ w/ Hessian + selective pruning | `train_gpt.py:1171-1224` |
| BPB | 1.11473 (3-seed mean) | submission.json |
| Step time | ~86.7ms | submission.json |
| Steps completed | ~6922 in 600s | submission.json |

---

## 3. Architecture Design

### 3.1 Layer Stack

```
Layer 0-11:  MambaBlock (12 layers)     — SSM recurrence, no attention
Layer 12-14: GQA Attention Block (3 layers) — full attention for long-range
Layer 15-17: MambaBlock (3 layers)      — final SSM refinement
```

Total: **18 layers** (15 Mamba + 3 Attention)

Alternative layout (interleaved):
```
[Mamba x4] → [Attn x1] → [Mamba x4] → [Attn x1] → [Mamba x4] → [Attn x1] → [Mamba x3]
```

### 3.2 MambaBlock Specification

```python
class MambaBlock(nn.Module):
    """Mamba-2 selective state-space layer with parallel associative scan."""

    # Dimensions
    d_model:  512          # input/output dimension
    d_state:  64           # SSM state dimension (N)
    d_conv:   4            # local convolution width
    expand:   2            # inner dimension = expand * d_model = 1024
    d_inner:  1024         # expanded dimension
    dt_rank:  32           # discretization timestep rank (d_model // 16)

    # Parameters per layer:
    #   in_proj:     512 x (1024*2 + 64 + 32) = 512 x 2120 = 1,085,440
    #      -> splits into: z (1024), x (1024), B (64), dt (32)
    #   conv1d:      d_inner x d_conv = 1024 x 4 = 4,096 (depthwise)
    #   x_proj:      NOT NEEDED (B, C, dt projected in in_proj for Mamba-2)
    #   dt_proj:     32 x 1024 (dt_rank -> d_inner) = 32,768
    #   A_log:       d_inner x d_state = 1024 x 64 = 65,536
    #   D:           d_inner = 1,024
    #   out_proj:    1024 x 512 = 524,288
    #
    # TOTAL per Mamba layer: ~1,713,152 (~1.71M)
    # Compare: Attention layer (Q+K+V+O+MLP) = ~2.62M
    # Savings: ~35% fewer params per layer
```

**Revised parameter budget with 15 Mamba + 3 Attention:**
| Component | Params | After Int6 (bytes) |
|-----------|--------|---------------------|
| 15 Mamba layers | 15 x 1.71M = 25.65M | ~19.2MB raw, ~14.4MB int6 |
| 3 GQA Attention layers | 3 x 2.62M = 7.86M | ~5.9MB raw, ~4.4MB int6 |
| Embedding (tied, 1024x512) | 0.52M | ~0.39MB |
| BigramHash 3072x112 | 0.34M + proj | ~0.3MB |
| Control tensors | ~0.05M | ~0.05MB |
| **Total** | **~34.4M** | **~19.5MB int6** |

**Problem:** 19.5MB > 16MB. Need to reduce.

### 3.3 Revised Design (Budget-Constrained)

To fit 16MB after int6+LZMA:

**Option A: Fewer Mamba layers (12 Mamba + 3 Attn = 15 total)**
| Component | Params | Int6 bytes |
|-----------|--------|------------|
| 12 Mamba (512d, d_state=64, expand=2) | 12 x 1.71M = 20.5M | ~11.5MB |
| 3 GQA Attn (512d, 8Q/4KV, 3x MLP) | 3 x 2.62M = 7.9M | ~4.4MB |
| Shared (embed, bigram, ctrl) | ~0.9M | ~0.7MB |
| **Total** | **~29.3M** | **~16.6MB raw → ~14.5MB LZMA** |

**Option B: Narrower Mamba (d_state=32, expand=1.5)**
| Component | Params | Per-layer |
|-----------|--------|-----------|
| in_proj: 512 x (768*2 + 32 + 32) = 512 x 1600 | 819K | |
| conv1d: 768 x 4 | 3K | |
| dt_proj: 32 x 768 | 24.6K | |
| A_log: 768 x 32 | 24.6K | |
| D: 768 | 0.8K | |
| out_proj: 768 x 512 | 393K | |
| **Per Mamba layer** | **~1.27M** | |
| 15 Mamba layers | 19.0M | |
| 3 GQA Attention layers | 7.9M | |
| Shared | 0.9M | |
| **Total** | **~27.8M → ~15.6MB int6 → ~13.5MB LZMA** |

**Recommended: Option B** — fits comfortably within 16MB with margin for tuning.

### 3.4 Attention Layers (Kept from SOTA)

The 3 attention layers use the EXACT same architecture as SOTA:
- `CausalSelfAttention` with GQA (8Q/4KV heads)
- XSA on all 3 attention layers
- Flash Attention 3 (FA3 Hopper kernels)
- RoPE with partial rope_dims=16
- QK normalization + q_gain
- `Block` wrapper with resid_mix, attn_scale, mlp_scale, ln_scale

### 3.5 U-Net Skip Connections

Maintain the encoder-decoder U-Net structure:
```
num_encoder_layers = 18 // 2 = 9
num_decoder_layers = 18 - 9 = 9
num_skip_weights = 9
```

Skip weights connect encoder layer i to decoder layer (17 - i).

---

## 4. New Code to Write

### 4.1 `MambaBlock` class (~120 lines)

**Insert location:** After `class SmearGate` (line 678), before `class BigramHashEmbedding` (line 680).

```python
class MambaBlock(nn.Module):
    """Mamba-2 selective state-space block with parallel scan."""
    def __init__(self, d_model=512, d_state=32, d_conv=4, expand=1.5):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        d_inner = int(expand * d_model)  # 768
        self.d_inner = d_inner
        dt_rank = max(d_model // 16, 1)  # 32
        self.dt_rank = dt_rank

        # Input projection: x, z (gate), B, dt all in one matmul
        self.in_proj = nn.Linear(d_model, d_inner * 2 + d_state + dt_rank, bias=False)

        # Depthwise conv (causal, applied to x branch only)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)

        # dt projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner, dtype=torch.float32))

        # C projection (from hidden state to output mixing)
        self.c_proj = nn.Linear(d_model, d_state, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.norm = RMSNorm()

    def forward(self, x):
        """
        x: (B, L, D)  ->  (B, L, D)
        """
        residual = x
        x = self.norm(x)

        # Project input
        proj = self.in_proj(x)  # (B, L, d_inner*2 + d_state + dt_rank)
        x_in, z, B, dt_in = proj.split(
            [self.d_inner, self.d_inner, self.d_state, self.dt_rank], dim=-1
        )

        # Causal conv1d on x branch
        x_in = x_in.transpose(1, 2)  # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :x.size(1)]  # causal trim
        x_in = x_in.transpose(1, 2)  # (B, L, d_inner)
        x_in = F.silu(x_in)

        # Compute C from original input
        C = self.c_proj(x)  # (B, L, d_state)

        # SSM computation
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dt = F.softplus(self.dt_proj(dt_in))  # (B, L, d_inner)

        # Discretize: dA = exp(A * dt), dB = dt * B
        # dt: (B, L, d_inner), A: (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Parallel associative scan (sequential fallback for correctness)
        # h[t] = dA[t] * h[t-1] + dB[t] * x[t]
        # y[t] = (C[t] @ h[t]) + D * x[t]
        B_size, L, d_inner = x_in.shape
        y = self._selective_scan(x_in, dA, dB, C, self.D)

        # Gate and project output
        y = y * F.silu(z)
        y = self.out_proj(y)

        return residual + y

    def _selective_scan(self, x, dA, dB, C, D):
        """
        Selective scan: h[t] = dA[t]*h[t-1] + dB[t]*x[t], y[t] = C[t]@h[t] + D*x[t]
        x: (B, L, d_inner)
        dA: (B, L, d_inner, d_state)
        dB: (B, L, d_inner, d_state)
        C: (B, L, d_state)
        D: (d_inner,)
        """
        B, L, d_inner = x.shape
        d_state = C.size(-1)

        # Use chunked parallel scan for efficiency
        # For now: sequential scan (replace with Triton kernel for speed)
        h = torch.zeros(B, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t, :, None]  # (B, d_inner, d_state)
            y_t = (h * C[:, t, None, :]).sum(-1) + D * x[:, t]  # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, d_inner)
```

**CRITICAL NOTE:** The sequential scan above is O(L * d_inner * d_state) per step and will be very slow. For production, replace with:
1. `mamba-ssm` package Triton/CUDA kernels (`pip install mamba-ssm`)
2. Or PyTorch-native parallel associative scan (`torch.cumsum` trick or custom Triton kernel)

### 4.2 `MambaHybridBlock` wrapper class (~30 lines)

Wraps either a MambaBlock or a standard attention Block with the same interface:

```python
class MambaHybridBlock(nn.Module):
    """Wrapper that dispatches to either MambaBlock or attention Block."""
    def __init__(self, is_mamba: bool, layer_idx: int, ...):
        super().__init__()
        self.is_mamba = is_mamba
        if is_mamba:
            self.mamba = MambaBlock(d_model, d_state, d_conv, expand)
        else:
            self.attn_block = Block(dim, num_heads, ...)
```

### 4.3 Modifications to `GPT.__init__` (lines 781-881)

**Changes needed:**
1. Accept `mamba_layers: list[int]` parameter specifying which layers are Mamba
2. Only allocate bank slots for attention layers (not Mamba layers)
3. Adjust `num_encoder_layers` / `num_decoder_layers` for 18-layer count
4. Create separate Mamba parameter list for optimizer

**Key change to parameter banks:**
```python
# Count only attention layers for banks
n_attn = sum(1 for i in range(num_layers) if i not in mamba_layer_set)
n_mamba = num_layers - n_attn

# Banks only for attention layers
self.qo_bank = nn.Parameter(torch.empty(2 * n_attn, model_dim, model_dim))
self.kv_bank = nn.Parameter(torch.empty(2 * n_attn, kv_dim, model_dim))
self.mlp_up_bank = nn.Parameter(torch.empty(n_attn, mlp_dim, model_dim))
self.mlp_down_bank = nn.Parameter(torch.empty(n_attn, model_dim, mlp_dim))

# Mamba layers have their own params (not banked)
self.mamba_blocks = nn.ModuleList([
    MambaBlock(d_model=model_dim, d_state=d_state, d_conv=d_conv, expand=expand)
    for _ in range(n_mamba)
])
```

### 4.4 Modifications to `GPT.forward` (lines 914-970)

**Changes needed:**
- Dispatch to either Mamba or attention based on layer type
- Mamba layers get `x` input only (no bank weights)
- Attention layers get bank weights as before

```python
def forward(self, input_ids, target_ids):
    # ... embedding unchanged ...
    attn_idx = 0  # tracks position in bank arrays
    for i in range(self.num_encoder_layers):
        if i in self.mamba_layer_set:
            x = self.mamba_blocks[self.mamba_idx_map[i]](x)
        else:
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[attn_idx](x, x0,
                self.qo_bank[attn_idx], ...)
            attn_idx += 1
        skips.append(x)
    # ... decoder similarly ...
```

### 4.5 Modifications to optimizer setup (lines 1665-1738)

**Changes needed:**
- Mamba `in_proj`, `out_proj`, `dt_proj` → Muon (2D matrix params)
- Mamba `A_log`, `D`, `conv1d` → Adam (scalar/1D params)
- Attention banks → Muon (unchanged)

```python
# Mamba 2D weights for Muon
mamba_matrix_params = []
for mb in base_model.mamba_blocks:
    mamba_matrix_params.extend([
        mb.in_proj.weight, mb.out_proj.weight,
        mb.dt_proj.weight, mb.c_proj.weight,
    ])

# Mamba scalar/1D params for Adam
mamba_scalar_params = []
for mb in base_model.mamba_blocks:
    mamba_scalar_params.extend([mb.A_log, mb.D, mb.dt_proj.bias])
    mamba_scalar_params.extend(list(mb.conv1d.parameters()))
```

**Note:** Mamba matrices are NOT banked (each layer has different shapes/purposes). They go through standard Muon with individual reduce-scatter, or just all-reduce + local NS5.

### 4.6 Modifications to GPTQ quantization (lines 1140-1313)

**Changes needed:**
- `_unbank_state_dict`: Only processes attention bank params (skip Mamba params)
- `_HessianGPT`: Add Mamba layers with CastedLinear wrappers for Hessian collection
- `mixed_quantize_int6`: Handle Mamba weight names (`mamba_blocks.N.in_proj.weight`, etc.)
- `_classify_param`: Add `"mamba"` category

```python
def _classify_param(name):
    if "mamba_blocks" in name:
        return "mamba"
    # ... existing logic ...

# In mixed_quantize_int6, add "mamba" to int6_cats:
quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn", "mamba"}, ...)
```

---

## 5. Hyperparameter Changes

Add to `Hyperparameters` class (line 28):

```python
# Mamba hybrid configuration
mamba_layers = os.environ.get("MAMBA_LAYERS", "0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17")
mamba_d_state = int(os.environ.get("MAMBA_D_STATE", 32))
mamba_d_conv = int(os.environ.get("MAMBA_D_CONV", 4))
mamba_expand = float(os.environ.get("MAMBA_EXPAND", 1.5))
num_layers = int(os.environ.get("NUM_LAYERS", 18))  # up from 11
```

---

## 6. Dependencies

```
# requirements.txt additions
mamba-ssm>=2.2.0     # For CUDA/Triton selective scan kernels
causal-conv1d>=1.4.0 # Efficient causal depthwise conv
```

**Fallback:** If `mamba-ssm` CUDA kernels don't work on H100/Hopper, use pure PyTorch sequential scan (slow but correct) or write a custom Triton kernel.

---

## 7. Parameter Budget Calculation

### Option B (Recommended: d_state=32, expand=1.5, 15 Mamba + 3 Attn)

| Component | Params | Int6 bytes | LZMA est |
|-----------|--------|------------|----------|
| 15x MambaBlock.in_proj (512 x 1600) | 12.29M | 9.22M | ~7.0M |
| 15x MambaBlock.out_proj (768 x 512) | 5.90M | 4.42M | ~3.4M |
| 15x MambaBlock.dt_proj (32 x 768) | 0.37M | fp16 | ~0.3M |
| 15x MambaBlock.c_proj (512 x 32) | 0.25M | fp16 | ~0.2M |
| 15x MambaBlock.A_log (768 x 32) | 0.37M | fp32 ctrl | ~0.3M |
| 15x MambaBlock.D (768) | 0.01M | fp32 ctrl | ~0.01M |
| 15x MambaBlock.conv1d (768 x 4) | 0.05M | fp16 | ~0.04M |
| 3x Attn qo_bank [6, 512, 512] | 1.57M | 1.18M | ~0.9M |
| 3x Attn kv_bank [6, 256, 512] | 0.79M | 0.59M | ~0.4M |
| 3x MLP up_bank [3, 1536, 512] | 2.36M | 1.77M | ~1.3M |
| 3x MLP down_bank [3, 512, 1536] | 2.36M | 1.77M | ~1.3M |
| tok_emb (1024 x 512, tied) | 0.52M | fp16 | ~0.4M |
| BigramHash 3072x112 + proj | 0.41M | fp16 | ~0.3M |
| Control tensors | ~0.08M | fp32 | ~0.06M |
| **Total** | **~27.3M** | **~21.0M raw** | **~15.6M LZMA** |

With code (~60KB): **~15.7MB total** — fits within 16MB.

---

## 8. Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `train_batch_tokens` | 786,432 | Same as SOTA |
| `train_seq_len` | 2048 | Same as SOTA |
| `max_wallclock_seconds` | 600 | Competition limit |
| `warmdown_iters` | 3500 | Same as SOTA |
| `matrix_lr` (Muon, attn) | 0.025 | Same as SOTA |
| `matrix_lr` (Muon, mamba) | 0.015 | Lower for Mamba (different landscape) |
| `scalar_lr` | 0.025 | Same as SOTA |
| `embed_lr` | 0.035 (tied) | Same as SOTA |
| `muon_momentum` | 0.99 | Same as SOTA |
| `ema_decay` | 0.997 | Same as SOTA |
| `swa_every` | 50 | Same as SOTA |
| `late_qat_threshold` | 0.15 | Same as SOTA |

---

## 9. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Mamba sequential scan too slow on H100 | HIGH | MEDIUM | Use mamba-ssm CUDA kernels; fall back to chunked scan |
| mamba-ssm package incompatible with PyTorch 2.9+ / CUDA 12.8 | HIGH | LOW | Pin versions; write custom Triton kernel if needed |
| Mamba underperforms attention at 512d scale | MEDIUM | MEDIUM | Keep 3 attention layers for long-range; ablate ratio |
| Training instability with mixed Mamba/Attn | MEDIUM | LOW | Separate LR groups; careful init |
| Step time > 100ms → fewer total steps | HIGH | MEDIUM | Profile early; reduce Mamba layers if needed |
| GPTQ doesn't work well on Mamba weights | LOW | LOW | Different weight distributions; ablate clip ranges |

---

## 10. Ablation Plan (Single-Seed, 1xH100)

| ID | Experiment | Expected Time |
|----|-----------|---------------|
| A1.1 | 18L all-Mamba (no attention) vs 15M+3A hybrid | 2x 10min |
| A1.2 | Attention layers: 1 vs 2 vs 3 vs 5 | 4x 10min |
| A1.3 | Mamba d_state: 16 vs 32 vs 64 | 3x 10min |
| A1.4 | Mamba expand: 1.0 vs 1.5 vs 2.0 | 3x 10min |
| A1.5 | Mamba position: bottom vs top vs interleaved | 3x 10min |
| A1.6 | Total layers: 12 vs 15 vs 18 vs 20 | 4x 10min |

---

## 11. Go/No-Go Checkpoints

| Checkpoint | Criteria | Fallback |
|-----------|----------|----------|
| Day 1: Mamba smoke test | Step time ≤ 100ms on 1xH100 | Abort arch1, focus arch2/3 |
| Day 3: Training convergence | Loss decreasing smoothly by step 1000 | Debug init/LR; reduce Mamba layers |
| Day 5: Full 8xH100 run | val_bpb ≤ 1.120 (pre-quant) | Reduce to fewer Mamba layers |
| Day 7: 3-seed eval | mean BPB ≤ 1.115 (post-quant) | Submit best of arch2/3 instead |

---

## 12. Files Modified (Diff Summary)

```
train_gpt.py
├── Hyperparameters: +6 new params (mamba_layers, d_state, d_conv, expand, num_layers=18)
├── +NEW class MambaBlock (~120 lines)
├── +NEW class MambaHybridBlock (~30 lines)  [optional wrapper]
├── GPT.__init__: banks sized for n_attn only; add mamba_blocks ModuleList
├── GPT.forward: dispatch loop (mamba vs attn)
├── GPT.forward_logits: same dispatch logic
├── _unbank_state_dict: handle mamba params (passthrough)
├── _rebank_state_dict: handle mamba params (passthrough)
├── _HessianGPT: add mamba layers with CastedLinear
├── _classify_param: add "mamba" category
├── main(): optimizer split for mamba params; mamba_matrix_params list
└── requirements.txt: +mamba-ssm, +causal-conv1d
```

Estimated total diff: **+250 lines, ~40 lines modified**.
