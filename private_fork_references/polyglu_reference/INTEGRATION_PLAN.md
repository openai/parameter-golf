# Integration Plan: PolyGLU into Parameter Golf

This is the step-by-step execution plan for Claude Code. Start from the best existing submission as a base, then integrate PolyGLU.

---

## Phase 0: Setup and Base Selection

1. **Start from the current SOTA submission** as the base. The leading submission (1.1194 BPB) is at:
   ```
   records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py
   ```
   Or use the cleanest high-scoring submission that's easiest to modify.

2. **Verify the base runs** and reproduces approximately the claimed score before making changes.

---

## Phase 1: Implement PolyMLP

### Step 1.1: Add the PolyMLP class

Replace the `MLP` class (or add alongside it) with `PolyMLP`. The implementation must be **torch.compile compatible** (no lambda closures in forward, no list comprehension with function calls).

```python
class PolyMLP(nn.Module):
    """
    PolyGLU-adapted MLP for Parameter Golf.
    Drop-in replacement for MLP with per-neuron activation routing.
    """
    def __init__(self, dim: int, mlp_mult: float, n_activations: int = 4):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.n_activations = n_activations
        self.hidden = hidden
        
        # Static routing preferences: [hidden, K]
        # Init zeros = uniform prior. MUST be exempt from weight decay.
        self.alpha = nn.Parameter(torch.zeros(hidden, n_activations))
        
        # Per-activation dynamic scaling
        self.beta = nn.Parameter(torch.ones(n_activations))
        
        # Dynamic gate network: dim → 16 → K
        self.gate_w1 = nn.Linear(dim, 16)
        self.gate_w2 = nn.Linear(16, n_activations)
        
        # Gumbel-Softmax temperature (updated externally)
        self.tau = 1.0

    def forward(self, x: Tensor) -> Tensor:
        # Routing computation
        h_mean = x.mean(dim=1)  # [B, dim]
        gate_hidden = torch.relu(self.gate_w1(h_mean))
        gate_out = self.gate_w2(gate_hidden)  # [B, K]
        
        logits = self.alpha.unsqueeze(0) + self.beta * gate_out.unsqueeze(1)
        # logits: [B, hidden, K]
        
        g_k = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        g_k = g_k.unsqueeze(1)  # [B, 1, hidden, K]
        
        # MLP forward
        h = self.fc(x)  # [B, seq, hidden]
        
        # Apply all K activations (torch.compile compatible — no lambdas)
        a0 = torch.relu(h).square()     # relu²
        a1 = torch.tanh(h)              # tanh
        a2 = F.silu(h)                  # SiLU
        a3 = F.gelu(h)                  # GELU
        activated = torch.stack([a0, a1, a2, a3], dim=-1)  # [B, seq, hidden, K]
        
        # Weighted activation selection
        h = (g_k * activated).sum(dim=-1)  # [B, seq, hidden]
        
        return self.proj(h)
```

### Step 1.2: Mark routing parameters for special handling

Add a name pattern so routing params are identified:

```python
# Add to the CONTROL_TENSOR_NAME_PATTERNS or create a new pattern set
ROUTING_PARAM_PATTERNS = ("alpha", "beta", "gate_w1", "gate_w2")
```

### Step 1.3: Update the Block class

Replace `MLP(dim, mlp_mult)` with `PolyMLP(dim, mlp_mult)` in the Block constructor:

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        # ... same attention setup ...
        self.mlp = PolyMLP(dim, mlp_mult)  # <-- Changed from MLP
        # ... same scales and resid_mix ...
```

The `forward` method of `Block` does NOT need to change — it just calls `self.mlp(self.mlp_norm(x))`.

---

## Phase 2: Optimizer Integration

### Step 2.1: Route PolyGLU parameters correctly

In the optimizer setup section of `main()`, ensure routing params go to Adam (not Muon):

```python
# When collecting matrix_params and scalar_params from block_named_params:
# alpha, beta, gate_w1.bias, gate_w2.bias → scalar_params (Adam)
# gate_w1.weight, gate_w2.weight → could go to either, but Adam is safer for small params

# The key check: alpha and beta should match scalar_params criteria.
# alpha is [hidden, K] which is 2D — it would normally go to matrix_params (Muon).
# Override: add "alpha" and "beta" to CONTROL_TENSOR_NAME_PATTERNS so they go to scalar_params.
```

Concrete modification to the CONTROL_TENSOR_NAME_PATTERNS:

```python
# EXISTING:
CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "skip_weight", "q_gain")

# MODIFIED — add routing params:
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "skip_weight", "q_gain",
    "alpha", "beta", "gate_w1", "gate_w2"
)
```

This ensures:
- α (2D) goes to scalar_params → Adam (not Muon)
- β (1D) goes to scalar_params → Adam
- gate_w1, gate_w2 (small) go to scalar_params → Adam

### Step 2.2: Exempt routing params from weight decay

Check the weight decay implementation. In the current codebase, weight decay is handled by the Muon optimizer's `muon_wd` parameter and/or Adam's `adam_wd`. Since routing params are now in the Adam/scalar group, ensure they are NOT subject to weight decay.

If the scalar optimizer applies weight decay, create a separate group:

```python
# Separate routing params from other scalars
routing_params = [
    p for name, p in block_named_params
    if any(pattern in name for pattern in ("alpha", "beta", "gate_w1", "gate_w2"))
]
other_scalar_params = [
    p for name, p in block_named_params
    if (p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    and not any(pattern in name for pattern in ("alpha", "beta", "gate_w1", "gate_w2"))
]

# Adam for routing params — NO weight decay
optimizer_routing = torch.optim.Adam(
    [{"params": routing_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": 0.0}],
    betas=(args.beta1, args.beta2),
    eps=args.adam_eps,
    fused=True,
)
optimizers.append(optimizer_routing)
```

---

## Phase 3: Temperature Annealing

### Step 3.1: Add tau annealing to the training loop

After the optimizer step in the main training loop, add tau update:

```python
# In the main training loop, after optimizer steps:
# Anneal Gumbel-Softmax temperature from 1.0 to 0.1
tau = max(0.1, 1.0 - 0.9 * step / args.iterations)
for block in base_model.blocks:
    if hasattr(block.mlp, 'tau'):
        block.mlp.tau = tau
```

### Step 3.2: Consider wall-clock-aware annealing

Since training stops on wall clock (not step count), the tau schedule should adapt:

```python
# Wall-clock aware tau annealing
if max_wallclock_ms is not None:
    progress = min(elapsed_ms / max_wallclock_ms, 1.0)
else:
    progress = step / args.iterations
tau = max(0.1, 1.0 - 0.9 * progress)
for block in base_model.blocks:
    if hasattr(block.mlp, 'tau'):
        block.mlp.tau = tau
```

---

## Phase 4: Quantization Compatibility

### Step 4.1: Verify routing params are handled correctly

The existing quantization pipeline keeps small tensors (< 65,536 elements) as fp16 passthrough. All routing params fall under this threshold:

- alpha: hidden × K = 1536 × 4 = 6,144 ✓
- beta: 4 ✓
- gate_w1.weight: 16 × 512 = 8,192 ✓
- gate_w1.bias: 16 ✓
- gate_w2.weight: 4 × 16 = 64 ✓
- gate_w2.bias: 4 ✓

No changes needed to `quantize_state_dict_int8` or `dequantize_state_dict_int8`.

### Step 4.2: Verify artifact size

After training, check that the total artifact (int8 + zlib/zstd + code) stays under 16MB. The routing overhead per layer is ~14.4K params × 2 bytes (fp16) = ~28.8KB. For 11 layers: ~317KB before compression. This is negligible vs. the 16MB budget.

---

## Phase 5: Evaluation and Tuning

### Step 5.1: At eval time, set tau to minimum

```python
# Before evaluation:
for block in base_model.blocks:
    if hasattr(block.mlp, 'tau'):
        block.mlp.tau = 0.1  # Near-deterministic routing at eval
```

### Step 5.2: Activation palette tuning

The initial palette {relu², tanh, SiLU, GELU} is a reasonable starting point. Consider these variants:

**Option A: Include LeakyReLU² (proven better in SOTA)**
```python
a0 = F.leaky_relu(h, negative_slope=0.5).square()  # LeakyReLU(0.5)²
a1 = torch.tanh(h)
a2 = F.silu(h)
a3 = F.gelu(h)
```

**Option B: All squared variants**
```python
a0 = torch.relu(h).square()
a1 = F.leaky_relu(h, 0.5).square()
a2 = F.silu(h).square()
a3 = F.gelu(h).square()
```

**Option C: K=3 to reduce overhead**
```python
a0 = F.leaky_relu(h, 0.5).square()  # Best known
a1 = torch.tanh(h)                   # Bounded (paper's deep-layer choice)
a2 = F.gelu(h)                       # Paper's early-layer choice
```

**Option D: K=2 minimal**
```python
a0 = F.leaky_relu(h, 0.5).square()  # Best known
a1 = torch.tanh(h)                   # Bounded compression
```

### Step 5.3: Hyperparameter sensitivity

Key hyperparameters to tune:
- `n_activations` (K): 2, 3, or 4
- `gate_bottleneck`: 8, 16, or 32
- `tau_min`: 0.1 (standard) or 0.05 (harder commitment)
- Whether to include α (full) or go dynamic-only
- Scalar LR for routing params: might benefit from higher LR than other scalars

---

## Phase 6: Advanced Optimizations (if time permits)

### 6.1: Per-layer activation profile (from paper findings)

The paper found early layers prefer GELU, deep layers prefer Tanh. You could bias the initialization:

```python
# In GPT.__init__, after creating blocks:
for i, block in enumerate(self.blocks):
    if hasattr(block.mlp, 'alpha'):
        depth_ratio = i / (num_layers - 1)  # 0.0 for first, 1.0 for last
        # Bias early layers toward GELU (k=3), deep layers toward Tanh (k=1)
        block.mlp.alpha.data[:, 3] += 0.5 * (1 - depth_ratio)  # GELU bias
        block.mlp.alpha.data[:, 1] += 0.5 * depth_ratio         # Tanh bias
```

### 6.2: Frozen routing after warmdown

During the warmdown phase, freeze routing (set tau very low and stop gradient on routing params):

```python
if in_warmdown_phase:
    for block in base_model.blocks:
        block.mlp.tau = 0.01  # Very hard routing
        block.mlp.alpha.requires_grad_(False)
        block.mlp.beta.requires_grad_(False)
        block.mlp.gate_w1.requires_grad_(False)
        block.mlp.gate_w2.requires_grad_(False)
```

### 6.3: Test-time routing distillation

After training, analyze which activation each neuron selected (argmax of routing), then create a simpler model with fixed per-neuron activations. This eliminates the routing computation entirely at eval time. However, this requires careful implementation and may not be worth the complexity.

---

## Checklist

- [ ] PolyMLP class implemented and torch.compile compatible
- [ ] Routing params routed to Adam optimizer (not Muon)
- [ ] Routing params exempt from weight decay
- [ ] Tau annealing added to training loop (wall-clock aware)
- [ ] Tau set to 0.1 at evaluation time
- [ ] Artifact size verified under 16MB
- [ ] Base score reproduced without PolyMLP (sanity check)
- [ ] Score with PolyMLP measured and compared to base
- [ ] Activation palette experimented with (at least 2 variants)
- [ ] Gate bottleneck size experimented with
- [ ] K value experimented with (2, 3, 4)
