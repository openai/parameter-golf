# PolyGLU Implementation Reference

This file contains the exact, annotated PolyGLU code from the original repository, plus the adapted version for Parameter Golf.

---

## Original PolyGLU (from PolychromaticLM — 600M params)

Source: `src/model/architecture.py` from https://github.com/danielxmed/PolyGLU

```python
import torch
import torch.nn as nn
import math


class PolyGLU(nn.Module):
    """
    PolyGLU: Polychromatic Gated Linear Unit.
    
    Drop-in replacement for SwiGLU. Each FFN neuron dynamically routes among K=4
    activation functions via a differentiable Gumbel-Softmax mechanism.
    
    Routing = static preference (α) + dynamic gating (β · gate_net(mean_pool(x)))
    
    Args:
        d_model: Hidden dimension (input/output size)
        d_ff: FFN intermediate dimension (number of neurons)
        n_activations: Number of candidate activation functions (K)
    """
    def __init__(self, d_model, d_ff, n_activations=4):
        super().__init__()

        # Standard SwiGLU projections — same shapes as vanilla SwiGLU
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)

        # Activation palette: K=4 qualitatively different nonlinearities
        self.activations = [
            nn.functional.relu,   # k=0: Hard threshold (Glutamate)
            torch.tanh,           # k=1: Symmetric compression (GABA)
            nn.functional.silu,   # k=2: Self-gated / Swish (Dopamine)
            nn.functional.gelu    # k=3: Probabilistic gate (Acetylcholine)
        ]

        # STATIC ROUTING: Per-neuron preference vector over K activations
        # Shape: [d_ff, K] = [4096, 4]
        # Initialized to ZERO → uniform prior (no initial preference)
        # IMPORTANT: Exempt from weight decay!
        self.alpha = nn.Parameter(torch.zeros(d_ff, n_activations))

        # DYNAMIC SCALING: Per-activation scaling for the dynamic signal
        # Shape: [K] = [4]
        # Initialized to ONE → dynamic signal fully active from start
        self.beta = nn.Parameter(torch.ones(n_activations))

        # Gumbel-Softmax temperature — updated externally by model.update_tau()
        self.tau = 1.0

        # DYNAMIC ROUTING: Lightweight MLP (d_model → 32 → K)
        # Input: mean-pooled hidden state (averages over sequence dimension)
        # This is the key component — paper shows it alone suffices for routing
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 32),   # 32 is a fixed bottleneck dim
            nn.ReLU(),
            nn.Linear(32, n_activations)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]

        # Step 1: Compute context vector via mean pooling over sequence
        mean_pool_h = torch.mean(x, 1)  # [batch, d_model]

        # Step 2: Compute routing logits = static + dynamic
        # alpha:    [d_ff, K] → unsqueeze(0) → [1, d_ff, K]
        # gate_net: [batch, K] → unsqueeze(1) → [batch, 1, K]
        # beta:     [K] broadcasts naturally
        # Result:   [batch, d_ff, K] — per-neuron, per-sample routing logits
        logits = self.alpha.unsqueeze(0) + (self.beta * self.gate_net(mean_pool_h).unsqueeze(1))

        # Step 3: Gumbel-Softmax to get differentiable routing weights
        # During training (tau=1.0→0.1): soft → near-hard selection
        # g_k shape: [batch, 1, d_ff, K] (unsqueeze for broadcasting with seq dim)
        g_k = nn.functional.gumbel_softmax(logits, tau=self.tau).unsqueeze(1)

        # Step 4: Standard SwiGLU-style projections
        gate_x = self.W_gate(x)  # [batch, seq_len, d_ff]
        value_x = self.W_up(x)   # [batch, seq_len, d_ff]

        # Step 5: Apply ALL K activations to the gate projection
        # nt_out shape: [batch, seq_len, d_ff, K]
        nt_out = torch.stack([nt(gate_x) for nt in self.activations], dim=-1)

        # Step 6: Weighted sum of activations per neuron
        # g_k: [batch, 1, d_ff, K] broadcasts over seq_len
        # Result: [batch, seq_len, d_ff]
        polyglu_sum = (g_k * nt_out).sum(dim=-1)

        # Step 7: Element-wise gate (same as SwiGLU's gate mechanism)
        polyglu_output = polyglu_sum * value_x

        # Step 8: Down-projection
        polyglu_output = self.W_down(polyglu_output)

        return polyglu_output
```

### Temperature Annealing (from PolychromaticLM)

```python
class PolychromaticLM(nn.Module):
    # ... (other methods) ...

    def update_tau(self, step, total_steps):
        """Linear annealing: τ goes from 1.0 to 0.1 over training."""
        tau_max = 1.0
        tau_min = 0.1
        tau = tau_max - (tau_max - tau_min) * (step / total_steps)
        tau = max(tau, tau_min)
        for block in self.model_core:
            block.polyglu.tau = tau
```

### TransformerBlock Integration

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_dim, seq_length, n_activations, eps,
                 n_q_heads, n_kv_heads, d_ff):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(d_model, eps)
        self.rmsnorm_2 = RMSNorm(d_model, eps)
        self.polyglu = PolyGLU(d_model, d_ff, n_activations)
        self.gqa = GQA(d_model, n_q_heads, n_kv_heads, eps)

    def forward(self, x, rope):
        # Pre-norm residual pattern (same as standard transformer)
        x_1 = self.rmsnorm_1(x)
        x_2 = self.gqa(x_1, rope)
        x_3 = x + x_2                    # Attention residual
        x_4 = self.rmsnorm_2(x_3)
        x_5 = self.polyglu(x_4)          # PolyGLU replaces SwiGLU here
        output = x_5 + x_3               # FFN residual
        return output
```

### Weight Initialization

```python
class PolychromaticLM(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def __init__(self, ...):
        # ...
        self.apply(self._init_weights)

        # Residual scaling for attention and FFN outputs
        scale = 1.0 / math.sqrt(2 * n_layers)
        for block in self.model_core:
            block.gqa.W_o.weight.data.mul_(scale)
            block.polyglu.W_down.weight.data.mul_(scale)
```

---

## Adapted PolyGLU for Parameter Golf

The Parameter Golf model uses a different MLP architecture than SwiGLU. The baseline uses:

```python
class MLP(nn.Module):
    # relu^2 MLP from the baseline
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim  # e.g., 2*512=1024 or 3*512=1536
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())  # relu² activation
```

Here is the adapted PolyGLU for this architecture:

```python
class PolyMLP(nn.Module):
    """
    PolyGLU-adapted MLP for Parameter Golf.
    
    Replaces the fixed relu² activation with per-neuron activation routing.
    Uses a simpler architecture than the full PolyGLU (no gate/up split)
    since the baseline MLP doesn't use gated projections.
    
    Design choices for Parameter Golf:
    - K=4 activations including relu² (the proven baseline)
    - Tiny gate network: dim→16→K (smaller than original's dim→32→K)
    - Static preferences (alpha) + dynamic gating
    - Same Gumbel-Softmax mechanism
    """
    def __init__(self, dim: int, mlp_mult: int, n_activations: int = 4):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        
        self.n_activations = n_activations
        self.hidden = hidden
        
        # Activation palette — includes relu² since it's the proven baseline
        # These are applied to the fc output before proj
        self.activation_fns = [
            lambda x: torch.relu(x).square(),           # k=0: relu² (baseline)
            lambda x: torch.tanh(x),                     # k=1: tanh (bounded)
            lambda x: torch.nn.functional.silu(x),       # k=2: SiLU/Swish
            lambda x: torch.nn.functional.gelu(x),       # k=3: GELU
        ]
        
        # Static routing preferences: [hidden_dim, K], init zeros (uniform prior)
        # EXEMPT FROM WEIGHT DECAY AND MUON OPTIMIZER
        self.alpha = nn.Parameter(torch.zeros(hidden, n_activations))
        
        # Dynamic scaling per activation
        self.beta = nn.Parameter(torch.ones(n_activations))
        
        # Dynamic gate network: tiny MLP
        # dim→16→K (smaller bottleneck for the small model)
        gate_bottleneck = 16
        self.gate_net = nn.Sequential(
            nn.Linear(dim, gate_bottleneck),
            nn.ReLU(),
            nn.Linear(gate_bottleneck, n_activations)
        )
        
        # Temperature for Gumbel-Softmax — managed externally
        self.tau = 1.0
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq_len, dim]
        
        # Compute context for routing (mean pool over sequence)
        h_mean = x.mean(dim=1)  # [batch, dim]
        
        # Routing logits: static + dynamic
        # alpha: [hidden, K], gate_net(h_mean): [batch, K]
        logits = self.alpha.unsqueeze(0) + self.beta * self.gate_net(h_mean).unsqueeze(1)
        # logits: [batch, hidden, K]
        
        # Gumbel-Softmax routing weights
        g_k = F.gumbel_softmax(logits, tau=self.tau, hard=False)  # [batch, hidden, K]
        g_k = g_k.unsqueeze(1)  # [batch, 1, hidden, K] for broadcasting over seq_len
        
        # Standard forward through fc
        h = self.fc(x)  # [batch, seq_len, hidden]
        
        # Apply all K activations
        activated = torch.stack([fn(h) for fn in self.activation_fns], dim=-1)
        # activated: [batch, seq_len, hidden, K]
        
        # Weighted combination
        h = (g_k * activated).sum(dim=-1)  # [batch, seq_len, hidden]
        
        # Down-projection
        return self.proj(h)
```

### Alternative: Simplified Dynamic-Only Routing

Since the paper shows the dynamic pathway alone suffices, we can drop α for an even leaner version:

```python
class PolyMLPDynamic(nn.Module):
    """
    Dynamic-only PolyGLU routing (no static α).
    Even fewer parameters — the gate network alone handles routing.
    """
    def __init__(self, dim: int, mlp_mult: int, n_activations: int = 4):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        
        self.activation_fns = [
            lambda x: torch.relu(x).square(),
            lambda x: torch.tanh(x),
            lambda x: F.silu(x),
            lambda x: F.gelu(x),
        ]
        
        # Dynamic-only: just the gate network, no α
        gate_bottleneck = 16
        self.gate_net = nn.Sequential(
            nn.Linear(dim, gate_bottleneck),
            nn.ReLU(),
            nn.Linear(gate_bottleneck, n_activations)
        )
        self.tau = 1.0
    
    def forward(self, x: Tensor) -> Tensor:
        h_mean = x.mean(dim=1)
        logits = self.gate_net(h_mean)  # [batch, K]
        # Expand to per-neuron: all neurons in a layer share the same routing
        # This is a simplification — original PolyGLU has per-neuron routing
        g_k = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        g_k = g_k.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, K]
        
        h = self.fc(x)
        activated = torch.stack([fn(h) for fn in self.activation_fns], dim=-1)
        h = (g_k * activated).sum(dim=-1)
        
        return self.proj(h)
```

### Parameter Overhead Calculation (for Parameter Golf)

For dim=512, hidden=1536 (3x MLP), K=4:

**Full PolyMLP (with α):**
- α: 1536 × 4 = 6,144 params
- β: 4 params
- gate_net: Linear(512→16) + Linear(16→4) = 8,192+16+64+4 = 8,276 params
- Total per layer: ~14,424 params
- Total for 11 layers: ~158,664 params (~0.6KB per layer in int8)

**Dynamic-only PolyMLPDynamic (no α):**
- gate_net: 8,276 params per layer
- Total for 11 layers: ~91,036 params

Both are well within the 16MB budget after quantization.

---

## Key Implementation Notes

1. **Gumbel-Softmax `hard=False`**: Use soft (continuous) routing during training. The routing naturally converges to near-hard selections via temperature annealing.

2. **Temperature schedule**: Anneal linearly from 1.0 to 0.1. For 10-minute training (~4000-5000 steps), this means roughly:
   ```python
   tau = max(0.1, 1.0 - 0.9 * step / total_steps)
   ```

3. **No auxiliary losses needed**: Do NOT add any sparsity loss, entropy penalty, or load-balancing loss. The paper proves the routing converges purely from the language modeling loss.

4. **Optimizer grouping**: Route α and β to the scalar/Adam optimizer group (NOT Muon). Exempt them from weight decay. The gate_net weights can go with the standard matrix/scalar splits.

5. **Quantization compatibility**: At checkpoint time, α, β, and gate_net weights are small tensors that will be kept as float16 passthrough (under the INT8_KEEP_FLOAT_MAX_NUMEL threshold). No special quantization handling needed.

6. **torch.compile compatibility**: The `torch.stack([fn(h) for fn in self.activation_fns], dim=-1)` pattern may need adjustment for torch.compile. A workaround:
   ```python
   # Instead of list comprehension, compute directly:
   relu2 = torch.relu(h).square()
   tanh_h = torch.tanh(h)
   silu_h = F.silu(h)
   gelu_h = F.gelu(h)
   activated = torch.stack([relu2, tanh_h, silu_h, gelu_h], dim=-1)
   ```
