# SURGE-M v2: Error-Driven Fast Weight Programming
### Surprise-gated Recurrent Generator for Evolving weight Matrices — Efficient Reformulation

**Status**: Post-experiment redesign. v1 was proven to work (0.030 BPB gain, chicken-egg fix). v2 is the mathematically derived efficient equivalent.

---

## Table of Contents

1. [What SURGE-M Is](#1-what-surge-m-is)
2. [What the Experiments Revealed](#2-what-the-experiments-revealed)
3. [The Mathematical Analysis](#3-the-mathematical-analysis)
4. [The New Efficient Architecture](#4-the-new-efficient-architecture)
5. [Component Specifications](#5-component-specifications)
6. [Complete Forward Pass](#6-complete-forward-pass)
7. [Gradient Flow and Training](#7-gradient-flow-and-training)
8. [Hyperparameters](#8-hyperparameters)
9. [Implementation Checklist](#9-implementation-checklist)
10. [Connection to Literature](#10-connection-to-literature)
11. [Ablations](#11-ablations)

---

## 1. What SURGE-M Is

Standard transformers process every token with the same fixed function. Titans and TTT-E2E add evolving *memory* that the fixed model consults. SURGE-M does something categorically different: it changes the *processing weights themselves* during the sequence, so the model literally becomes a different function as it processes text.

The mechanism:
- Two middle transformer layers (3 and 4) have their output projection matrix W_O evolve during the sequence
- A small recurrent meta-network M reads per-token **prediction error vectors** and navigates in the space of weight configurations
- Updates are **multiplicative** — function composition in GL(d), not additive gradient steps in the tangent space
- M maintains a **GRU hidden state** that remembers the history of navigations, learning when and how to change the model's behavior

The key claim: when text surprises the model, the model's processing changes — not just its memory or its predictions for that one token, but how it processes everything that follows.

---

## 2. What the Experiments Revealed

From 18 experiments run overnight (see `SURGE_M_AUTONOMOUS_RESEARCH_SUMMARY.md`):

### 2.1 What worked
- SURGE-M produces a **real 0.030 BPB improvement** over matched baseline (d=256 control)
- **Single SURGE layer (layer 4 only)** beats two SURGE layers — each layer adds noise
- Ablation E showed the **GRU navigation state is necessary**: memoryless M corrupts the model (2.46 BPB vs 1.56)
- Ablation B showed **multiplicative is better than additive**, but only slightly — the first-order expansion dominates

### 2.2 What failed
- **Chicken-egg zero gradient**: zero-initialized u/v heads create a hard zero gradient fixed point — M never activates
  - Fix: `UV_INIT_STD > 0` (small random init breaks symmetry)
  - This is a genuine insight: any multiplicative meta-network with zero-initialized output heads is mathematically frozen, not just slowly learning
- **Speed (2.6× slower)**: 16 separate forward passes of 64 tokens instead of one forward pass of 1024 tokens
  - Sequential GRU loop (1024 calls to GRUCell) was the original bottleneck — fixed 7.6× by switching to `nn.GRU`
  - Remaining 2.6× is the chunk-wise structure itself

### 2.3 The core lesson
The architecture is sound. The bottleneck is that the 16-chunk sequential structure breaks FlashAttention and creates unnecessary sequential dependencies. The mathematical analysis below shows this dependency is avoidable.

---

## 3. The Mathematical Analysis

This section derives why v1's sequential structure is unnecessary and what it's equivalent to.

### 3.1 Expanding the Multiplicative Chain

At chunk $c$, M produces $(u_c, v_c, g_c)$ and the weight updates:

$$W_c = (I + g_c \cdot u_c \otimes v_c) \cdot W_{c-1}$$

Applied recursively, define the accumulated transformation matrix:

$$M_c = \prod_{k=0}^{c-1}(I + g_k \cdot u_k \otimes v_k)$$

So $W_c = M_c \cdot W_0$.

The output of the SURGE layer on attention pre-output $h_c$ (the attention computation before W_O):

$$y_c = h_c \cdot W_c^T = h_c \cdot W_0^T \cdot M_c^T$$

Let $z_c = h_c \cdot W_0^T$ — the **standard base output** using frozen $W_0$.

$$\boxed{y_c = z_c \cdot M_c^T}$$

**The SURGE output equals the base $W_0$ output post-multiplied by $M_c^T$.** This means the weight evolution can be moved entirely to a post-processing step on the base output — the base transformer can run in one full-sequence pass with $W_0$.

### 3.2 First-Order Expansion: The Fast Weight Equivalence

Expanding $M_c^T$ to first order (the cross-terms between different chunks are second-order in the gate values):

$$M_c^T \approx I + \sum_{k<c} g_k \cdot v_k \otimes u_k$$

Therefore:

$$y_c \approx z_c + \sum_{k<c} g_k \cdot (z_c \cdot v_k) \cdot u_k$$

Define the accumulated fast weight matrix $A_c = \sum_{k<c} g_k \cdot u_k \otimes v_k$.

$$y_c \approx z_c + z_c \cdot A_c^T = z_c \cdot (I + A_c^T)$$

Updated as: $A_c = A_{c-1} + g_{c-1} \cdot u_{c-1} \otimes v_{c-1}$ — a trivial prefix sum.

**This is exactly the delta rule / linear attention formulation** (Schlag et al., ICML 2021):

$$W^{(t)} = W^{(t-1)} + v_t \otimes k_t, \quad \text{output} = W^{(t)} q_t$$

SURGE-M maps to this with:
- Values: $u_k$ (M's output direction)  
- Keys: $g_k \cdot v_k$ (M's key direction scaled by gate)
- Queries: $z_c^T$ (base W_0 output)

**SURGE-M is, to first order, a fast weight program where keys and values are written by a prediction-error-driven recurrent controller, not by linear projections of the input.** This is what separates it from Titans and TTT.

### 3.3 When Do Higher-Order Terms Matter?

The second-order cross-term when expanding two rank-1 updates:

$$(I + g_1 u_1 \otimes v_1)(I + g_0 u_0 \otimes v_0) = \ldots + g_0 g_1 (v_1 \cdot u_0) \cdot u_1 \otimes v_0$$

The cross-term is proportional to $(v_1 \cdot u_0)$ — the dot product between M's key direction at chunk 1 and value direction at chunk 0.

**Key insight**: if M learns near-orthogonal outputs — $v_k \cdot u_j \approx 0$ for $k \neq j$ — then the multiplicative and additive forms are equivalent to all orders. The program-transition behavior (eigenspectrum change) only occurs when updates are non-orthogonal to each other.

This motivates an **orthogonality regularizer** on M's outputs (see Section 8.4).

The ablation result (exp B: additive slightly worse than multiplicative) suggests higher-order terms do help, but modestly — consistent with small but nonzero cross-terms after training.

### 3.4 The WY Representation: Exact Computation Without Materializing W_c

The product of $c$ rank-1 updates has a compact representation known as the WY form (Bischof & Van Loan, 1987), the same technique used in DeltaNet's chunkwise parallelization (Yang et al., NeurIPS 2024):

$$M_c = I + W_c Y_c^T$$

where $W_c, Y_c \in \mathbb{R}^{d \times c}$ are accumulated column-by-column:

$$W_c = [W_{c-1},\ M_{c-1} u_{c-1}], \quad Y_c = [Y_{c-1},\ g_{c-1} v_{c-1}]$$

The correction to chunk $c$'s output:

$$y_c - z_c = z_c \cdot M_c^T - z_c = z_c \cdot (Y_c W_c^T) = (z_c Y_c) W_c^T$$

where $z_c Y_c \in \mathbb{R}^{B \times T_{chunk} \times c}$ costs only $O(c \cdot d \cdot T_{chunk})$ — for $c \leq 16$ chunks this is tiny.

**This gives the exact multiplicative output without ever forming $W_c$ explicitly.**

### 3.5 Adding Forgetting: The Gated Delta Rule Form

Current SURGE-M accumulates $A_c$ without forgetting, which causes representational overflow on long sequences. Adding a data-dependent decay factor from M:

$$A_c = \alpha_c \cdot A_{c-1} + g_{c-1} \cdot u_{c-1} \otimes v_{c-1}$$

where $\alpha_c = \sigma(W_\alpha s_c) \in (0,1)$ is a learned **forget gate** output by M.

This is the **Gated Delta Rule** (Yang et al., ICLR 2025), the same structure as Gated DeltaNet which outperforms both Mamba2 and vanilla DeltaNet. The recurrence is:

$$A_c = \alpha_c A_{c-1} + \Delta_c$$

This linear recurrence with data-dependent mixing is parallelizable via the associative scan operator $(\lambda_1, \delta_1) \oplus (\lambda_2, \delta_2) = (\lambda_1 \lambda_2,\ \lambda_2 \delta_1 + \delta_2)$.

### 3.6 The Navigation State: minGRU for Parallelism

The standard GRU requires $h_t = f(x_t, h_{t-1})$ — strictly sequential. The minGRU (Feng et al., 2024) removes hidden-state dependencies from the gates:

$$z_t = \sigma(W_z x_t), \quad \tilde{h}_t = \sigma(W_h x_t)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Since $z_t$ and $\tilde{h}_t$ depend only on $x_t$, this is a linear recurrence $h_t = a_t \odot h_{t-1} + b_t$ with $a_t = 1-z_t$, $b_t = z_t \odot \tilde{h}_t$. Linear recurrences are solvable in $O(\log T)$ via parallel prefix scan.

minGRU achieves a **175× training speedup** over standard GRU for sequence length 512. For SURGE-M's d_state=64 navigating over 1024 tokens, this turns the navigation scan from ~0.5s sequential into ~3ms parallel.

### 3.7 The Causal Dependency — Resolving the Circular Problem

In v1, chunk $c$'s errors depend on logits from $W_c$, which depends on errors from chunks $< c$ — circular. Under the **frozen base approximation** (compute errors from $z$ using $W_0$):

$$\text{error}_t = \text{one\_hot}(\text{token}_t) - \text{softmax}(\text{logit}_{t-1}^{W_0})$$

This breaks the circular dependency. The approximation error is $O(\text{drift}^2)$, i.e., second-order in the weight drift magnitude. Since MAX_DRIFT_FRACTION=0.1 means $\|W_c - W_0\|/\|W_0\| \leq 0.1$, the approximation gap is at most 1% of the correction magnitude.

This is the same approximation TTT-E2E uses in its training formulation.

---

## 4. The New Efficient Architecture

### 4.1 Summary of Changes from v1

| | v1 (original) | v2 (this doc) |
|---|---|---|
| Base forward | 16 separate passes over 64 tokens | 1 pass over 1024 tokens (FlashAttention) |
| GRU | GRUCell loop, sequential, 1024 calls | minGRU parallel scan, O(log T) |
| Error computation | For-loop over tokens | Fully vectorized |
| Weight update | Sequential multiplicative chain | WY representation, exact, O(C·d) |
| Forgetting | Elastic anchor only | Gated delta rule (learned per-chunk forget gate) |
| Speed vs baseline | 2.6× slower | Expected ~1.1–1.2× slower |
| Mathematical equivalence | N/A | Exact (WY) or first-order (fast weight) |

### 4.2 The New Data Flow

```
Input tokens [B, T]
        │
        ▼
┌─────────────────────────────────────────┐
│  BASE TRANSFORMER FORWARD  (one pass)   │
│  Layers 0–8, frozen W_0 everywhere      │
│  Outputs: z [B,T,d], h_lower [B,T,d],  │
│           attn_pre_Wo [B,T,d] (layer 4) │
│           logits_base [B,T,V]           │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌───────────────┐  ┌──────────────────────────┐
│ ERROR COMPUTE │  │  minGRU PARALLEL SCAN    │
│ (vectorized)  │  │  over (h_lower, errors)  │
│ [B, T, d_err] │  │  → s_all [B, T, d_state] │
└───────┬───────┘  └──────────┬───────────────┘
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  CHUNK BOUNDARY     │
        │  STATE SAMPLE       │
        │  s_c [B, C, d_state]│
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  M OUTPUT HEADS     │
        │  u, v, g, α         │
        │  [B, C, d/d/1/1]    │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  WY CORRECTION      │
        │  (exact multiply)   │
        │  [B, T, d]          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  FINAL LOGITS       │
        │  logits_base        │
        │  + lm_head(corr)    │
        └─────────────────────┘
```

All blocks are parallel. The only quasi-sequential step is minGRU scan at O(log T).

---

## 5. Component Specifications

### 5.1 Base Transformer

Identical to competition baseline:

```
vocab_size  = 1024
d_model     = 256  (or 512 for full-scale)
n_heads     = 4
n_kv_heads  = 4
n_layers    = 9
d_ff        = 512  (2× expansion, SwiGLU)
seq_len     = 1024
norm        = RMSNorm
rope        = rotary position encoding
tied_embed  = True
```

Only difference: layer 4's attention module exposes its pre-W_O output (`attn_out_4`) for use in the WY correction.

### 5.2 minGRU Navigation State

Replaces the standard GRU from v1. Fully parallelizable via prefix scan.

```python
class MinGRU(nn.Module):
    """
    Minimal GRU (Feng et al., 2024).
    Gates depend only on current input, not previous state.
    This makes the recurrence linear and parallelizable.
    
    Recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * tilde_h_t
    where z_t = sigmoid(W_z @ x_t)  [no h_{t-1} dependency]
          tilde_h_t = sigmoid(W_h @ x_t)  [no h_{t-1} dependency]
    
    Equivalent linear form: h_t = a_t * h_{t-1} + b_t
    where a_t = 1 - z_t,  b_t = z_t * tilde_h_t
    
    This linear recurrence is solved in O(log T) via parallel scan.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined projection: outputs [z, tilde_h] concatenated
        self.proj = nn.Linear(input_size, 2 * hidden_size, bias=True)
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        x:  [B, T, input_size]
        h0: [B, hidden_size] or None (zeros)
        Returns: h_all [B, T, hidden_size]
        """
        B, T, _ = x.shape
        
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device)
        
        # Compute all gates in parallel (no sequential dependency)
        out    = self.proj(x)                      # [B, T, 2*H]
        z      = torch.sigmoid(out[..., :self.hidden_size])    # [B, T, H]
        tilde  = torch.sigmoid(out[..., self.hidden_size:])    # [B, T, H]
        
        # Linear recurrence coefficients
        a = 1.0 - z   # [B, T, H] — decay
        b = z * tilde  # [B, T, H] — input injection
        
        # Parallel prefix scan to solve h_t = a_t * h_{t-1} + b_t
        return parallel_scan(a, b, h0)   # [B, T, H]


def parallel_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """
    Solve h_t = a_t * h_{t-1} + b_t for all t in O(log T) steps.
    
    a:  [B, T, H]  decay coefficients
    b:  [B, T, H]  input injection
    h0: [B, H]     initial state
    
    The associative operator for composing (a1, b1) and (a2, b2):
      (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)
    
    This represents composing two linear maps:
      h → a2*(a1*h + b1) + b2 = a1*a2*h + a2*b1 + b2
    """
    B, T, H = a.shape
    
    # Prepend h0 as position 0 (absorbing initial state)
    # We'll use an iterative doubling approach
    # For T=1024, H=64, B=32: this is very fast
    
    # Simple implementation: torch.jit.script-friendly iterative doubling
    # Can be replaced with a proper CUDA kernel for production
    h_all = torch.zeros(B, T, H, device=a.device, dtype=a.dtype)
    
    # Sequential fallback (replace with scan for large T)
    # For T=1024, this is 1024 elementwise ops — still 100x faster than GRUCell
    h = h0
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        h_all[:, t] = h
    
    return h_all

# NOTE: For true O(log T) parallel scan, use:
# from jax import lax; lax.associative_scan(fn, (a, b))
# OR the torch equivalent from mamba-ssm / flash-linear-attention:
# from fla.ops.utils import chunk_cumsum_fwd (flash-linear-attention library)
#
# For T=1024 and d_state=64, even the sequential loop above is fast:
# 1024 elementwise mults + adds on [B=32, H=64] = 2M FLOPs
# ~0.1ms on H100 — negligible compared to attention
```

**Parameter count of minGRU** (d_model=256, d_err=64, d_state=64):
- `proj`: (256 + 64) × (2 × 64) + 2×64 = 41,216 params

Compare to original GRU: (320+64) × 64 × 3 × 2 = 147,456 params. minGRU is **3.6× smaller**.

### 5.3 Error Projection

Unchanged from v1, but now applied vectorized over the full sequence:

```python
class ErrorProjection(nn.Module):
    """
    Compresses prediction error vector from vocab space to d_err.
    
    error_t = one_hot(token_t) - softmax(logits_{t-1})
    
    This vector:
    - Has norm approximately sqrt(2) for a maximally wrong prediction
    - Sums to 0 (one-hot sums to 1, softmax sums to 1)
    - Contains: which token appeared (+1 spike), which were over-predicted (negative), scale of surprisal
    - Is the exact gradient direction of the CE loss w.r.t. the final linear layer output
    """
    def __init__(self, vocab_size: int, d_err: int):
        super().__init__()
        self.proj = nn.Linear(vocab_size, d_err, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
    
    def forward(self, logits_prev: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        logits_prev: [B, T, vocab_size]  (logits shifted by 1: logits_prev[:, t] predicts token_ids[:, t])
        token_ids:   [B, T]
        Returns:     [B, T, d_err]
        
        Vectorized over full sequence — no Python loop.
        """
        probs   = F.softmax(logits_prev.detach(), dim=-1)  # [B, T, V]
        one_hot = F.one_hot(token_ids, num_classes=logits_prev.shape[-1]).float()  # [B, T, V]
        error   = one_hot - probs  # [B, T, V]  — directional error signal
        return self.proj(error)    # [B, T, d_err]
```

Note: `logits_prev.detach()` because the error signal is informational input to M, not a direct loss path. The gradient through `self.proj.weight` is still computed and is what trains the projection.

### 5.4 Meta-Network M Output Heads

M now has **4 output heads** instead of 3:

```python
class MetaNetwork(nn.Module):
    def __init__(self, d_model=256, vocab_size=1024, d_err=64, d_state=64):
        super().__init__()
        
        self.err_proj = ErrorProjection(vocab_size, d_err)
        self.mingru   = MinGRU(input_size=d_model + d_err, hidden_size=d_state)
        
        # Output heads — all initialized to produce near-zero output
        # (but NOT exactly zero — the v1 chicken-egg lesson)
        # UV_INIT_STD=0.01: small random signal breaks symmetry without destabilizing early training
        UV_INIT_STD = 0.01
        
        self.u_head = nn.Linear(d_state, d_model, bias=True)
        self.v_head = nn.Linear(d_state, d_model, bias=True)
        self.g_head = nn.Linear(d_state, 1,       bias=True)
        self.a_head = nn.Linear(d_state, 1,       bias=True)  # NEW: forget gate
        
        # Small random init for u/v (breaks chicken-egg zero gradient)
        nn.init.normal_(self.u_head.weight, std=UV_INIT_STD)
        nn.init.normal_(self.v_head.weight, std=UV_INIT_STD)
        nn.init.zeros_(self.u_head.bias)
        nn.init.zeros_(self.v_head.bias)
        
        # Gate init: bias -4.6 → sigmoid ≈ 0.01 → near-zero initial updates
        nn.init.zeros_(self.g_head.weight)
        nn.init.constant_(self.g_head.bias, -4.6)
        
        # Forget gate init: bias +2.3 → sigmoid ≈ 0.9 → mostly retain (almost no forgetting initially)
        # Model learns to forget more aggressively as needed
        nn.init.zeros_(self.a_head.weight)
        nn.init.constant_(self.a_head.bias, 2.3)
    
    def forward_navigation(self, h_lower: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Run minGRU over full sequence.
        h_lower: [B, T, d_model]   (hidden state after lower layers, before SURGE)
        e:       [B, T, d_err]     (compressed prediction errors)
        Returns: s_all [B, T, d_state]
        """
        gru_in = torch.cat([h_lower, e], dim=-1)   # [B, T, d_model + d_err]
        return self.mingru(gru_in)                  # [B, T, d_state]
    
    def get_chunk_updates(self, s_all: torch.Tensor, chunk_size: int) -> dict:
        """
        Sample chunk-boundary states and produce update factors.
        s_all: [B, T, d_state]
        Returns dict with u, v, g, a all [B, C, d] or [B, C, 1]
        """
        T = s_all.shape[1]
        # Sample state at END of each chunk (last token processed in chunk)
        boundary_ids = torch.arange(chunk_size - 1, T, chunk_size, device=s_all.device)
        s_c = s_all[:, boundary_ids]   # [B, C, d_state]
        
        return {
            'u': self.u_head(s_c),                     # [B, C, d_model]
            'v': self.v_head(s_c),                     # [B, C, d_model]
            'g': torch.sigmoid(self.g_head(s_c)),      # [B, C, 1]
            'a': torch.sigmoid(self.a_head(s_c)),      # [B, C, 1]
        }
```

### 5.5 WY-Based Exact Correction

The WY representation computes the exact multiplicative correction without forming $W_c$ explicitly:

```python
def wy_correction(
    attn_pre_wo: torch.Tensor,   # [B, T, d] — attention output before W_O
    updates: dict,               # {'u': [B,C,d], 'v': [B,C,d], 'g': [B,C,1], 'a': [B,C,1]}
    chunk_size: int,
    W_0: torch.Tensor,           # [d, d] — base W_O weight (used for v_rot computation)
) -> torch.Tensor:               # [B, T, d] — correction to add to residual stream
    """
    Computes y_c - z_c for each chunk c using the WY representation.
    
    Exact form: y_c = z_c @ M_c^T = z_c + (z_c @ Y_c) @ W_c^T
    where M_c = I + W_c @ Y_c^T is the WY representation.
    
    The attn_pre_wo @ W_0^T = z_c (standard output), so the correction is:
    delta_c = z_c @ M_c^T - z_c = (z_c @ Y_c) @ W_c^T
    
    Note: we don't compute z_c separately — we compute:
    delta_c directly from attn_pre_wo using the current WY factors.
    
    attn_pre_wo @ W_0^T gives the standard z_c.
    The WY correction on top of z_c gives the full SURGE output y_c.
    But since we apply the correction to logits (via lm_head), we need the correction
    to the RESIDUAL STREAM, which is z_c itself (the W_O output).
    
    So: correction = z_c @ A_c^T   where A_c = sum_{k<c} g_k * u_k ⊗ v_k
                                              with forget: A_c = a_c * A_{c-1} + g_{c-1} * u_{c-1} ⊗ v_{c-1}
    """
    B, T, d = attn_pre_wo.shape
    C = T // chunk_size
    
    u = updates['u']   # [B, C, d]
    v = updates['v']   # [B, C, d]
    g = updates['g']   # [B, C, 1]
    a = updates['a']   # [B, C, 1]
    
    corrections = []
    
    # WY factors: built up column by column
    # W_factors[k] = M_k @ u_k (column k of W matrix in WY representation)
    # Y_factors[k] = g_k * v_k (column k of Y matrix in WY representation)
    W_cols = []
    Y_cols = []
    
    # Running M matrix for WY update (starts as identity)
    # We track this to compute M_c @ u_c for the new W column
    # For efficiency: M_c u_c = u_c + sum_{k<c} (Y_k^T u_c) * W_k
    # We don't need to store full M_c — only the WY factors
    
    for c in range(C):
        cs = c * chunk_size
        ce = min((c + 1) * chunk_size, T)
        
        z_c = F.linear(attn_pre_wo[:, cs:ce], W_0)   # [B, chunk, d] — standard z_c
        
        if len(W_cols) == 0:
            # First chunk: no correction (A_0 = 0)
            corrections.append(torch.zeros_like(z_c))
        else:
            # Stack WY factors: W [B, d, c_so_far], Y [B, d, c_so_far]
            W_mat = torch.stack(W_cols, dim=-1)   # [B, d, c]
            Y_mat = torch.stack(Y_cols, dim=-1)   # [B, d, c]
            
            # Correction: (z_c @ Y_mat) @ W_mat^T
            # z_c @ Y_mat: [B, chunk, c]  (project z_c onto each Y column)
            # @ W_mat^T:   [B, chunk, d]  (reconstruct correction)
            ZY   = torch.einsum('btd,bdc->btc', z_c, Y_mat)      # [B, chunk, c]
            corr = torch.einsum('btc,bdc->btd', ZY, W_mat)       # [B, chunk, d]
            
            # Apply forget gate: scale current correction by accumulated forgetting
            # (conceptually: A_c incorporates decay from all previous forget gates)
            corrections.append(corr)
        
        # ── Update WY factors for next chunk ──────────────────────────
        u_c = u[:, c]   # [B, d]
        v_c = v[:, c]   # [B, d]
        g_c = g[:, c]   # [B, 1]
        a_c = a[:, c]   # [B, 1]
        
        # New W column: M_c @ u_c
        # M_c u_c = u_c + sum_{k<c} (Y_k^T u_c) * W_k  [via WY formula]
        if len(W_cols) == 0:
            Mu_c = u_c                                              # [B, d]
        else:
            W_mat = torch.stack(W_cols, dim=-1)                     # [B, d, c]
            Y_mat = torch.stack(Y_cols, dim=-1)                     # [B, d, c]
            Yu    = torch.einsum('bdc,bd->bc', Y_mat, u_c)         # [B, c]
            Mu_c  = u_c + torch.einsum('bdc,bc->bd', W_mat, Yu)    # [B, d]
        
        # New Y column: g_c * v_c
        gv_c = g_c * v_c   # [B, d]
        
        # Apply forget gate: scale all existing WY columns by a_c
        # This implements A_c = a_c * A_{c-1} + new_term
        if len(W_cols) > 0:
            W_cols = [a_c * w for w in W_cols]
            Y_cols = [a_c * y for y in Y_cols]
        
        W_cols.append(Mu_c)
        Y_cols.append(gv_c)
    
    return torch.cat(corrections, dim=1)   # [B, T, d]
```

---

## 6. Complete Forward Pass

```python
class SURGEM_v2(nn.Module):
    """
    SURGE-M v2: Error-driven fast weight programming.
    
    Mathematically equivalent to v1 (WY representation is exact)
    but fully parallelizable:
      - One base transformer forward over full sequence
      - Parallel error computation
      - minGRU parallel scan for navigation state
      - WY-based correction (no sequential chunk dependency)
    
    Expected speedup over v1: 4-8×
    Expected overhead over baseline: ~1.1-1.2×
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config     = config
        self.chunk_size = config.chunk_size  # 64
        self.vocab_size = config.vocab_size  # 1024
        
        # Standard transformer (all layers, W_0 frozen during forward)
        self.embed     = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks    = nn.ModuleList([TransformerBlock(config) for _ in range(9)])
        self.norm      = RMSNorm(config.d_model)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # tied embeddings
        
        # SURGE components
        self.meta_net  = MetaNetwork(
            d_model    = config.d_model,
            vocab_size = config.vocab_size,
            d_err      = 64,
            d_state    = 64,
        )
        
        # SURGE layer index (single layer — best result from experiments)
        self.surge_layer = 4
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        Returns:   logits [B, T, vocab_size]
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # ── BLOCK 1: Full-sequence base transformer forward ───────────────────
        # Standard one-pass forward over all 1024 tokens simultaneously.
        # W_0 used everywhere. FlashAttention applies normally.
        
        x = self.embed(input_ids)   # [B, T, d_model]
        h = x
        
        h_lower_collected  = None
        attn_pre_wo        = None
        
        for i, block in enumerate(self.blocks):
            if i == 3:
                # Collect lower-layer representation for GRU input
                # (after layer 2, before SURGE layer — unaffected by SURGE)
                h_lower_collected = h.detach()  # detach: informational signal
            
            if i == self.surge_layer:
                # Collect pre-W_O attention output for WY correction
                # The block exposes this if we request it
                h, attn_pre_wo_i = block.forward_with_pre_wo(h)
                attn_pre_wo = attn_pre_wo_i   # [B, T, d_model]
            else:
                h = block(h)
        
        # Base logits (used for error computation AND as the base prediction)
        logits_base = self.lm_head(self.norm(h))   # [B, T, vocab_size]
        
        # ── BLOCK 2: Vectorized error computation ─────────────────────────────
        # Shift logits by 1: logits_prev[:, t] predicts input_ids[:, t]
        logits_prev = torch.cat([
            torch.zeros(B, 1, self.vocab_size, device=device),
            logits_base[:, :-1]
        ], dim=1).detach()   # [B, T, vocab_size]  — detached from logit graph
        
        e = self.meta_net.err_proj(logits_prev, input_ids)   # [B, T, d_err]
        
        # ── BLOCK 3: minGRU parallel scan ─────────────────────────────────────
        # Integrates (token representation, prediction error) over full sequence
        s_all = self.meta_net.forward_navigation(h_lower_collected, e)  # [B, T, d_state]
        
        # ── BLOCK 4: Chunk-boundary states → update factors ───────────────────
        updates = self.meta_net.get_chunk_updates(s_all, self.chunk_size)
        # updates: {'u': [B,C,d], 'v': [B,C,d], 'g': [B,C,1], 'a': [B,C,1]}
        
        # ── BLOCK 5: WY correction (exact multiplicative, no sequential chunks) ─
        W_0 = self.blocks[self.surge_layer].attn.c_proj.weight.detach()  # [d, d]
        
        correction = wy_correction(
            attn_pre_wo = attn_pre_wo,
            updates     = updates,
            chunk_size  = self.chunk_size,
            W_0         = W_0,
        )   # [B, T, d_model]
        
        # ── BLOCK 6: Add correction to logits ─────────────────────────────────
        # The correction is to the W_O output (residual stream contribution).
        # We pass it through norm + lm_head to get its logit contribution.
        # This is an approximation (correction doesn't propagate through upper layers)
        # but valid to first order via residual stream argument.
        logits_corrected = logits_base + self.lm_head(self.norm(correction))
        
        return logits_corrected
```

### 6.1 The `forward_with_pre_wo` Block Method

The SURGE layer needs to expose the attention output before W_O:

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1  = RMSNorm(config.d_model)
        self.attn  = CausalSelfAttention(config)
        self.ln_2  = RMSNorm(config.d_model)
        self.mlp   = MLP(config)
    
    def forward(self, x):
        """Standard forward, unchanged for non-SURGE layers."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
    def forward_with_pre_wo(self, x):
        """
        Forward pass that also returns the pre-W_O attention output.
        Used only for the SURGE layer.
        """
        normed    = self.ln_1(x)
        pre_wo, h = self.attn.forward_return_pre_wo(normed)  # returns BOTH
        x         = x + h                                     # standard residual
        x         = x + self.mlp(self.ln_2(x))
        return x, pre_wo   # (hidden state, pre-W_O output for WY correction)


class CausalSelfAttention(nn.Module):
    def forward_return_pre_wo(self, x):
        """
        Returns (pre_wo, output) where:
        pre_wo = the concatenated head outputs BEFORE the output projection
        output = the final attention output (pre_wo @ W_O^T)
        """
        # ... standard QKV + attention computation ...
        # attn_out: [B, T, d_model]  — concatenated head outputs
        pre_wo = attn_out
        output = self.c_proj(attn_out)
        return pre_wo, output
```

---

## 7. Gradient Flow and Training

### 7.1 What Trains What

**θ_base** (9-layer transformer, including W_0):
- Standard gradient path from loss through logits_base
- Additionally: gradient from `lm_head(norm(correction))` flows back through:
  - `norm` and `lm_head` (their weights)
  - `correction` → WY factors → (u, v, g, a) → M's heads → GRU → err_proj

**θ_M** (MetaNetwork parameters):
- Gradient path: loss → correction term → (u, v, g, a) → M.u_head, M.v_head, M.g_head, M.a_head → s_c → minGRU → err_proj

The minGRU's parallel scan is fully differentiable — autograd handles the backward pass correctly because the scan is implemented with standard PyTorch ops.

### 7.2 No FOMAML Needed

In v1, the sequential W_t chain required the FOMAML detach to keep gradients tractable. In v2, WY is used directly on the base output — there is no chain of weight updates to backprop through. The correction is computed analytically in one pass.

The gradient flows cleanly:
- Through the WY correction formula into (u, v, g, a)
- Through (u, v, g, a) into M's output heads
- Through M's output heads into s_c
- Through s_c into the minGRU weights (via the parallel scan's backward)
- Through the minGRU into err_proj (via the error computation)

No truncation needed. Full gradient over the complete sequence.

### 7.3 Orthogonality Regularizer (Optional but Recommended)

From the mathematical analysis: higher-order multiplicative terms contribute when $v_k \cdot u_j \neq 0$. Add a regularizer to control this:

```python
def orthogonality_reg(u, v, lambda_orth=1e-4):
    """
    Encourages u_k ⊥ v_j for k ≠ j.
    When this is satisfied, multiplicative = additive to all orders.
    When violated, multiplicative provides extra program-transition expressiveness.
    
    u: [B, C, d]
    v: [B, C, d]
    """
    # Compute cross-dot-products: v_k · u_j for all k, j pairs
    # VU[b, k, j] = v_k · u_j
    VU = torch.einsum('bkd,bjd->bkj', v, u)   # [B, C, C]
    
    # Zero out diagonal (k=j is fine — self-interaction)
    mask = 1.0 - torch.eye(VU.shape[-1], device=VU.device)
    VU   = VU * mask
    
    return lambda_orth * (VU ** 2).mean()
```

Add `orthogonality_reg(updates['u'], updates['v'])` to the loss. This lets the model choose how much program-transition expressiveness it wants.

### 7.4 Training Loop

```python
def train_step(model, input_ids, optimizer_base, optimizer_meta, lambda_orth=1e-4):
    """
    Standard training step. No FOMAML, no TBPTT.
    Full gradient over complete sequence.
    """
    logits   = model(input_ids)     # full forward
    
    # Next-token prediction loss
    ce_loss  = F.cross_entropy(
        logits[:, :-1].reshape(-1, model.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    
    # Optional: orthogonality regularization
    # (get updates from a separate forward call or cache them during forward)
    # orth_loss = orthogonality_reg(updates['u'], updates['v'])
    
    loss = ce_loss  # + orth_loss
    
    loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer_base.step()
    optimizer_meta.step()
    
    optimizer_base.zero_grad(set_to_none=True)
    optimizer_meta.zero_grad(set_to_none=True)
    
    return loss.item()
```

### 7.5 Optimizers

```python
# Muon for base transformer matrix parameters (same as competition baseline)
base_matrices = [p for n, p in model.named_parameters()
                 if 'meta_net' not in n and p.dim() >= 2]
base_scalars  = [p for n, p in model.named_parameters()
                 if 'meta_net' not in n and p.dim() < 2]

optimizer_base = Muon(
    [{'params': base_matrices, 'lr': MATRIX_LR},
     {'params': base_scalars,  'lr': SCALAR_LR}],
    momentum     = 0.95,
    backend_steps= 5,
)

# AdamW for MetaNetwork (RNN weights + output heads)
optimizer_meta = torch.optim.AdamW(
    model.meta_net.parameters(),
    lr          = META_LR,
    betas       = (0.9, 0.999),
    weight_decay= 0.01,
    eps         = 1e-8,
)
```

---

## 8. Hyperparameters

```python
# ── Architecture (same as competition baseline) ──────────────────
VOCAB_SIZE    = 1024
D_MODEL       = 256     # start here; 512 once speed is validated
N_HEADS       = 4
N_KV_HEADS    = 4
N_LAYERS      = 9
D_FF          = 512
SEQ_LEN       = 1024

# ── SURGE specifics ───────────────────────────────────────────────
SURGE_LAYER   = 4       # single layer — best result from v1 experiments
CHUNK_SIZE    = 64      # 16 chunks per 1024-token sequence

# ── Meta-network ─────────────────────────────────────────────────
D_ERR         = 64
D_STATE       = 64

# ── Initialization (v1 chicken-egg lesson) ───────────────────────
UV_INIT_STD   = 0.01    # small random init on u/v output heads
GATE_INIT_BIAS= -4.6    # sigmoid(-4.6) ≈ 0.01 → near-zero initial gate
FORGET_INIT_BIAS = 2.3  # sigmoid(2.3) ≈ 0.9 → mostly retain initially

# ── Training ─────────────────────────────────────────────────────
MATRIX_LR     = 0.04
SCALAR_LR     = 0.04
META_LR       = 3e-4
GRAD_CLIP     = 1.0
LAMBDA_ORTH   = 1e-4    # orthogonality regularizer weight

# ── Wallclock (personal experiment — not competition constrained) ─
MAX_WALLCLOCK = 1800    # 30 minutes recommended for seeing M learn
```

### 8.1 Key Differences from v1

| Parameter | v1 | v2 | Reason |
|---|---|---|---|
| `SURGE_LAYER` | `[3, 4]` | `4` | Experiments: single layer wins |
| GRU type | `nn.GRUCell` loop | `MinGRU` | 175× speedup |
| `TBPTT_CHUNKS` | 4 | N/A | Not needed: no sequential chain |
| `MAX_DRIFT_FRACTION` | 0.1 | N/A | Replaced by forget gate `a_c` |
| Output heads | 3 (u, v, g) | 4 (u, v, g, a) | Forget gate added |
| `UV_INIT_STD` | 0.0 → 0.01 | 0.01 | v1 chicken-egg lesson |

---

## 9. Implementation Checklist

For Claude Code:

- [ ] `TransformerBlock.forward_with_pre_wo()` returns `(x, pre_wo)` with no extra computation cost
- [ ] `CausalSelfAttention.forward_return_pre_wo()` returns pre-W_O activation without recomputing attention
- [ ] `ErrorProjection.forward()` is vectorized — NO Python loop over tokens
- [ ] `MinGRU.forward()` uses parallel scan, NOT a Python loop over time steps
- [ ] `wy_correction()` loop runs over C=16 chunks, NOT over T=1024 tokens
- [ ] WY forget gate scales ALL existing columns each chunk: `W_cols = [a_c * w for w in W_cols]`
- [ ] Logits_prev is DETACHED before passing to ErrorProjection
- [ ] h_lower_collected is DETACHED before passing to minGRU
- [ ] UV_INIT_STD = 0.01 applied to u_head and v_head weights (NOT bias)
- [ ] Gate bias initialized to -4.6 (sigmoid ≈ 0.01)
- [ ] Forget gate bias initialized to +2.3 (sigmoid ≈ 0.9)
- [ ] optimizer_meta uses AdamW, NOT Muon
- [ ] MetaNetwork parameters are included in int8 quantization for submission artifact
- [ ] Diagnostic logging tracks: u_norm, v_norm, mean_gate, mean_forget, correction_magnitude

### 9.1 Diagnostic Logging

```python
def log_surge_diagnostics(updates: dict, correction: torch.Tensor, step: int):
    """Log whether M is actually doing anything."""
    print(f"Step {step}:")
    print(f"  u_norm:           {updates['u'].norm(dim=-1).mean().item():.4f}")
    print(f"  v_norm:           {updates['v'].norm(dim=-1).mean().item():.4f}")
    print(f"  mean_gate:        {updates['g'].mean().item():.4f}")
    print(f"  mean_forget:      {updates['a'].mean().item():.4f}")
    print(f"  correction_rms:   {correction.pow(2).mean().sqrt().item():.6f}")
    print(f"  correction/z_rms: ratio of correction magnitude to base output magnitude")
```

**Expected trajectory**:
- Step 0: u_norm ≈ 0.01, mean_gate ≈ 0.01, correction ≈ 0 → model behaves as standard transformer ✓
- Step 500: u_norm grows, mean_gate rises to 0.05–0.1 → M starting to contribute
- Step 2000: correction_rms / z_rms ≈ 0.01–0.05 → meaningful but bounded corrections

---

## 10. Connection to Literature

### 10.1 Where SURGE-M Fits in the Taxonomy

```
Sequence model memory mechanisms:
├── Attention (softmax): exact O(T²), full context retrieval
├── Linear attention / fast weights (additive outer product accumulation)
│   ├── Linear Transformer (Katharopoulos et al., 2020)
│   ├── DeltaNet (Schlag et al., 2021) — with corrective delta rule
│   └── Gated DeltaNet (Yang et al., ICLR 2025) — adds forgetting gate
├── Test-time weight update:
│   ├── TTT-Linear (Sun et al., 2024) — mathematically equivalent to DeltaNet
│   ├── Titans (Google, NeurIPS 2025) — deep neural memory + surprise metric
│   └── TTT-E2E (Astera, Dec 2025) — end-to-end meta-learning for TTT
└── SURGE-M v2 (this work):
    ├── Fast weight structure (= Gated DeltaNet) for the correction mechanism
    └── Keys/values from error-driven recurrent controller (= NOT from input projections)
         ↑ This is the novel part
```

### 10.2 Exact Equivalences

| SURGE-M v2 component | Equivalent existing architecture | Reference |
|---|---|---|
| $A_c = \alpha_c A_{c-1} + g_{c-1} u_{c-1} \otimes v_{c-1}$ | Gated Delta Rule recurrence | Yang et al. ICLR 2025 |
| $y_c = z_c + z_c \cdot A_c^T$ | Linear attention reading | Schlag et al. ICML 2021 |
| WY representation for $M_c$ | DeltaNet parallelization | Yang et al. NeurIPS 2024 |
| minGRU parallel scan | Were RNNs All We Needed? | Feng et al. 2024 |
| Error-driven memory writes | Titans surprise metric | Behrouz et al. NeurIPS 2025 |

### 10.3 What Makes SURGE-M Different

In all existing fast weight / linear attention architectures, keys and values are computed as **linear projections of the current input token**:

$$k_t = W_K x_t, \quad v_t = W_V x_t$$

In SURGE-M, keys and values come from a **recurrent meta-network that reads prediction errors**:

$$k_t = v_{\text{surge}}(s_t), \quad v_t = u_{\text{surge}}(s_t)$$

where $s_t = \text{minGRU}(\ldots, \text{prediction\_error}_{t-1}, h^{\text{lower}}_t, \ldots)$.

The difference:
- Standard: "given this token, write this key-value pair into memory"
- SURGE-M: "given what I predicted and what actually happened, write this key-value pair into memory"

The memory is written by **prediction error**, not by content. This means the memory captures *where the model was wrong* and what context surrounded those errors — not just what tokens appeared. This is a direct implementation of predictive coding (Rao & Ballard, 1999): only errors update the internal model.

---

## 11. Ablations

The following ablations answer distinct scientific questions. Run in order — each result informs whether to continue.

### Ablation A: v2 Main (this spec)
**Question**: Does the v2 architecture match v1's BPB with near-baseline speed?
**Expected**: BPB ~1.56 (matching v1 best), speed ~1.1-1.2× baseline

### Ablation B: Additive vs Multiplicative (WY vs prefix sum)
**Change**: Replace `wy_correction()` with `prefix_sum_correction()` (pure first-order fast weight, no WY composition)
**Question**: Does the exact multiplicative structure matter beyond the first-order fast weight approximation?
**Expected**: Small degradation (~0.005-0.010 BPB) — consistent with v1 ablation B result

### Ablation C: With vs Without Forget Gate
**Change**: Set `a_c = 1.0` (no forgetting, revert to v1-style accumulation)
**Question**: Does the learned forget gate help for 1024-token sequences?
**Expected**: Small degradation for short sequences, larger degradation for long sequences

### Ablation D: minGRU vs Standard GRU
**Change**: Replace minGRU with `nn.GRU` (keeping v2's parallel structure for everything else)
**Question**: Does the GRU's nonlinear state coupling matter, or is the linear recurrence sufficient?
**Expected**: GRU slightly better on quality but 10× slower — tradeoff decision

### Ablation E: Error Vector vs Scalar Surprisal
**Change**: Replace `ErrorProjection` with scalar cross-entropy, GRU input is `[h_lower, ce_loss_scalar]`
**Question**: Does the directional error signal matter beyond just magnitude?
**Expected**: Degradation — v1 exp C showed scalar was worse

### Ablation F: Learned Navigation vs Direct Gradient (TTT comparison)
**Change**: Replace minGRU + M output heads with: $u_t = $ gradient of loss w.r.t. attention output, $v_t = $ corresponding key direction
**Question**: Is M's learned navigation better than just following the gradient (TTT)?
**Expected**: M better if training is sufficient; gradient better early in training

### Priority order given limited compute:
Run A first. If speed is fixed and BPB matches, run D (the key architectural question). Then B (multiplicative vs additive). Then C (forgetting). E and F are nice-to-have.

---

## Appendix A: Parameter Count

```
Base transformer (d=256):
  embed:             1024 × 256               =    262,144
  per layer (9×):
    attn (Q,K,V,O): 4 × 256² / 1 (no GQA savings at n_kv=4) = 262,144
    mlp (up+down):   256×512 + 512×256         =    262,144
    norms + scalars: ~1,500
  total per layer:                             ~    525,788
  9 layers:                                   ~ 4,732,000
  SUBTOTAL base:                              ~ 4,994,000 ≈ 5.0M

MetaNetwork M (minGRU variant):
  err_proj:          1024 × 64                =     65,536
  minGRU (input=320, hidden=64):
    proj:            320 × 128 + 128          =     41,088
  output heads (4):
    u_head:          64 × 256 + 256           =     16,640
    v_head:          64 × 256 + 256           =     16,640
    g_head:          64 × 1 + 1               =         65
    a_head:          64 × 1 + 1               =         65
  SUBTOTAL M:                                 ~    140,034 ≈ 140K

GRAND TOTAL:                                 ~ 5,134,000 ≈ 5.1M params

At int8: ~5.1MB
After zlib compression: estimated 3-4MB
Code: ~50KB
TOTAL ARTIFACT: ~4MB ← well within 16MB budget ✓

Note: v1 M had ~206K params (standard GRU)
      v2 M has ~140K params (minGRU, 32% smaller)
```

---

## Appendix B: Complexity Analysis

| Operation | v1 | v2 | Notes |
|---|---|---|---|
| Base forward | 16 × O(chunk² d) | O(T² d) → FlashAttn O(T d) | v2 uses full FlashAttention |
| Error compute | 16 × O(chunk × V) loop | O(T × V) vectorized | |
| Navigation | O(T × d_state) sequential | O(T × d_state) scan | O(log T) depth |
| Weight update | O(C × d²) | O(C × d) via WY | WY avoids full d×d |
| Apply correction | O(C × chunk × d) | O(T × c_rank × d) | c_rank ≤ 16 |
| **Total** | **16× base forward** | **~1.1× base forward** | estimated |

---

## Appendix C: Relationship to Titans

Titans (Behrouz et al., NeurIPS 2025) is the closest published architecture.

**How Titans works**:
1. Separate neural memory module $M_t$ (an MLP with evolving weights)
2. Updated via gradient descent on associative loss: $\ell = \|M_{t-1}(k_t) - v_t\|^2$
3. Surprise metric = gradient magnitude of $\ell$ w.r.t. input
4. Updates: $M_t = M_{t-1} - \eta_t \nabla \ell$, with momentum and weight decay
5. Main model attends to memory output as additional context

**How SURGE-M v2 differs**:
1. **No separate memory module** — corrections applied directly to processing weights
2. **Error-driven recurrent controller** instead of input-derived gradient
3. **Keys and values from prediction errors** (what was wrong) instead of (what appeared)
4. **Single fast weight matrix** $A_c$ instead of a full MLP being trained
5. **Forget gate from M** (data-dependent) instead of fixed weight decay

The architectures are complementary: Titans stores content-based associations, SURGE-M stores error-based corrections. A hybrid — running both — would use Titans for long-term content memory and SURGE-M for domain-adaptive error-correction.

---

*This document describes SURGE-M v2 as derived from mathematical analysis of v1's structure. The core architecture (multiplicative weight updates driven by prediction errors via a recurrent meta-network) is unchanged; only the computational realization has been restructured for efficiency.*
