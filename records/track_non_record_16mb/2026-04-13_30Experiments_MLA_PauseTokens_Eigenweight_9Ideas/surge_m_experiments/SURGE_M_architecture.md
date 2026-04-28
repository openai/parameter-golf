# SURGE-M Architecture Specification
## Surprise-gated Recurrent Generator for Evolving weight Matrices

---

## 1. Motivation and Core Insight

Standard transformers have **fixed weights during the forward pass**. Every token in every sequence is processed by the same function. This is the fundamental limitation identified as "model-blindness" — the data and model state are never jointly considered.

Recent work like Titans (Google, NeurIPS 2025) adds a *separate* neural memory module whose weights evolve. But the main processing weights stay fixed. The memory is auxiliary — it influences output but the core computation doesn't change.

TTT-E2E updates the model's own weights, but uses gradient descent (additive updates). Additive updates move W along the tangent space (Lie algebra). Small steps, continuous change, no phase transitions.

**SURGE-M does something categorically different**:

1. **The processing weights themselves evolve** — not auxiliary memory, the actual middle layers change
2. **Updates are multiplicative**: W_t = (I + gate * u ⊗ v) @ W_{t-1} — this is function composition, not addition
3. **A recurrent meta-network M decides the update** — not gradient following, but learned navigation in program space
4. **Input to M is the full prediction error vector** — directional information (what was wrong) not just scalar surprisal

### Why Multiplicative Matters

Additive: W_t = W_{t-1} + ΔW
- Moves in the Lie algebra (tangent space near identity)
- W changes continuously and proportionally
- The fixed points of the UT iteration shift continuously

Multiplicative: W_t = (I + ΔW) @ W_{t-1}
- Composes functions in the Lie group GL(d)
- Can change the eigenspectrum of W
- Potentially causes bifurcations: a small rank-1 multiplicative change can flip the fixed-point structure entirely
- This is what "moving between programs" means mathematically

### Why the Prediction Error Vector Matters

Scalar surprisal tells M: "you were wrong by X bits"
Error vector tells M: "you over-predicted these tokens, under-predicted this one"

The error vector at position t is:
```
error_t = softmax(logits_{t-1}) - one_hot(token_t)    # shape [vocab_size]
```

This is the exact gradient direction for the output layer. It contains:
- Which token actually appeared
- How confident the wrong predictions were
- The direction in output space that was violated

M receives this compressed to 64 dims and learns: "when predictions fail in THAT direction, move THESE weights THIS way."

---

## 2. Architecture Overview

```
Input tokens → Embedding (vocab=1024, d_model=256)
                    ↓
         Layers 0, 1, 2  (standard, fixed)
                    ↓
         SURGE Layer 3   (W_O evolves multiplicatively)
                    ↓
         SURGE Layer 4   (W_O evolves multiplicatively)
                    ↓
         Layers 5, 6, 7, 8  (standard, fixed)
                    ↓
         Output head (tied to embedding)
         
Plus (stored):
   Meta-network M  (GRU, d_state=64, ~170k params)
```

**SURGE layers 3 and 4**: These are standard transformer blocks with one modification — their output projection weight matrix W_O is not frozen. It starts as W_0 (base initialization) and is updated multiplicatively by M at each chunk boundary.

**Everything else**: Identical to the competition baseline (9 layers, d_model=256, 4 attention heads, 4 KV heads for GQA, d_ff=512, SentencePiece vocab=1024, tied embeddings, Muon optimizer).

---

## 3. Stored Parameters (Submission Artifact)

| Component | Size | Notes |
|---|---|---|
| θ_base | ~8.5M | Standard 9-layer transformer weights including W_0_3, W_0_4 |
| θ_M | ~170K | GRU weights + output heads + error projector |
| **Total** | **~8.67M** | Well within 16MB budget |

**Ephemeral state (reset each sequence, not stored)**:
- W_3_t, W_4_t: current SURGE weights (start at W_0_3, W_0_4, evolve during sequence)
- s_t: GRU navigation state (starts at zeros)

---

## 4. Meta-network M Architecture

```python
class MetaNetwork(nn.Module):
    def __init__(self, d_model=256, vocab_size=1024, d_err=64, d_state=64):
        super().__init__()
        
        # Compress prediction error from vocab space to d_err
        # Uses LayerNorm for stability (error vectors have high variance)
        self.err_proj = nn.Sequential(
            nn.Linear(vocab_size, d_err, bias=False),
            nn.LayerNorm(d_err)
        )
        
        # GRU: integrates (token_repr, error_signal) over time
        # Maintains navigation state in program space
        self.gru = nn.GRUCell(
            input_size  = d_model + d_err,
            hidden_size = d_state,
        )
        
        # Output heads for multiplicative update factors
        # ZERO INITIALIZED: at start of training, M produces no updates
        # Model degrades gracefully to standard transformer initially
        self.u_head  = nn.Linear(d_state, d_model, bias=False)
        self.v_head  = nn.Linear(d_state, d_model, bias=False)
        self.gate    = nn.Linear(d_state, 1, bias=True)
        
        # SEPARATE heads for each SURGE layer
        # Layer 3 and Layer 4 may need different updates
        self.u_head_3 = nn.Linear(d_state, d_model, bias=False)
        self.v_head_3 = nn.Linear(d_state, d_model, bias=False)
        self.u_head_4 = nn.Linear(d_state, d_model, bias=False)
        self.v_head_4 = nn.Linear(d_state, d_model, bias=False)
        self.gate_3   = nn.Linear(d_state, 1, bias=True)
        self.gate_4   = nn.Linear(d_state, 1, bias=True)
        
        # Zero init output heads
        for head in [self.u_head_3, self.v_head_3, self.u_head_4, self.v_head_4]:
            nn.init.zeros_(head.weight)
        # Gate biases: initialize to large negative → sigmoid ≈ 0.01
        nn.init.constant_(self.gate_3.bias, -4.6)
        nn.init.constant_(self.gate_4.bias, -4.6)
    
    def compress_error(self, logits_prev, token_ids):
        """
        Compute prediction error vector and compress it.
        
        logits_prev: [B, vocab_size] — logits BEFORE seeing token_ids
        token_ids: [B] — the actual tokens that appeared
        
        Returns: [B, d_err] — compressed directional error signal
        """
        probs = F.softmax(logits_prev, dim=-1)  # [B, vocab]
        one_hot = F.one_hot(token_ids, num_classes=logits_prev.shape[-1]).float()
        error = one_hot - probs  # [B, vocab] — signed error, sums to 0
        return self.err_proj(error)  # [B, d_err]
    
    def update_state(self, h_lower_t, e_t, s_prev):
        """
        Update GRU navigation state for one token.
        
        h_lower_t: [B, d_model] — lower layers output for this token
        e_t: [B, d_err] — compressed prediction error (from PREVIOUS token)
        s_prev: [B, d_state] — previous navigation state
        
        Returns: s_t [B, d_state]
        """
        gru_input = torch.cat([h_lower_t, e_t], dim=-1)  # [B, d_model + d_err]
        return self.gru(gru_input, s_prev)
    
    def get_update(self, s):
        """
        Produce multiplicative update factors from current navigation state.
        Called once per chunk boundary.
        
        s: [B, d_state]
        
        Returns: (u3, v3, gate3, u4, v4, gate4) — update factors for both SURGE layers
        """
        u3    = self.u_head_3(s)                    # [B, d_model]
        v3    = self.v_head_3(s)                    # [B, d_model]
        gate3 = torch.sigmoid(self.gate_3(s))       # [B, 1]
        
        u4    = self.u_head_4(s)                    # [B, d_model]
        v4    = self.v_head_4(s)                    # [B, d_model]
        gate4 = torch.sigmoid(self.gate_4(s))       # [B, 1]
        
        return u3, v3, gate3, u4, v4, gate4
```

---

## 5. Multiplicative Weight Update

Given update factors (u, v, gate) for a SURGE layer:

```python
def apply_multiplicative_update(W_prev, u, v, gate, W_0, max_drift_fraction=0.1):
    """
    Apply: W_t = (I + gate * u ⊗ v) @ W_prev
    
    W_prev: [d, d] — current weight matrix (W_O of SURGE layer)
    u: [B, d] — update direction (left factor)
    v: [B, d] — update direction (right factor)
    gate: [B, 1] — how much to update (close to 0 = no update)
    W_0: [d, d] — original base weight (for drift constraint)
    
    Note: For batch processing, u and v are averaged over batch:
    u_mean = u.mean(0), v_mean = v.mean(0), gate_mean = gate.mean(0)
    
    This makes W_t the same for all sequences in the batch.
    (Could also do per-sequence W if memory allows, but expensive)
    """
    u_mean    = u.mean(0)           # [d]
    v_mean    = v.mean(0)           # [d]
    gate_mean = gate.mean(0).item() # scalar
    
    d = W_prev.shape[0]
    
    # Efficient computation: W_new = W_prev + gate * outer(u, v) @ W_prev
    # = W_prev + gate * u * (v^T @ W_prev)
    # Avoids materializing the full I + gate * outer(u,v) matrix
    v_W = v_mean @ W_prev           # [d] — v^T W_prev
    delta = gate_mean * torch.outer(u_mean, v_W)  # [d, d] rank-1
    W_new = W_prev + delta
    
    # Elastic anchor: constrain drift from base
    drift = (W_new - W_0).norm(p='fro')
    max_drift = max_drift_fraction * W_0.norm(p='fro')
    if drift > max_drift:
        # Project W_new back to ball of radius max_drift around W_0
        excess = drift / max_drift
        W_new = W_0 + (W_new - W_0) / excess
    
    return W_new.detach()  # DETACH: W_t is not part of the backward graph
                           # (FOMAML: don't backprop through the chain of updates)
```

**Important**: The DETACH on the returned W_new is the FOMAML approximation. Gradients flow FROM the loss THROUGH W_t BACKWARD INTO the (u, v, gate) that produced it. But they don't flow through W_{t-1} → (u_{prev}, v_{prev}) → M_prev. This truncates the meta-gradient to one update step.

---

## 6. Full Forward Pass

```python
class SURGE_M_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed     = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc   = # rotary or learned positional encoding
        
        # 9 transformer layers
        self.layers    = nn.ModuleList([
            TransformerBlock(config) for _ in range(9)
        ])
        # Mark which layers are SURGE layers
        self.surge_layer_indices = [3, 4]
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_head.weight = self.embed.weight  # tie embeddings
        
        self.meta_net = MetaNetwork(
            d_model    = config.d_model,
            vocab_size = config.vocab_size,
            d_err      = 64,
            d_state    = 64,
        )
        
        self.chunk_size = 64
    
    def forward(self, input_ids):
        """
        Full sequence forward pass with chunk-wise weight evolution.
        
        input_ids: [B, T]
        Returns: logits [B, T, vocab_size]
        """
        B, T = input_ids.shape
        
        # Embed tokens
        x = self.embed(input_ids)  # [B, T, d_model]
        
        # Initialize ephemeral state
        W_3 = dict(self.layers[3].named_parameters())   # snapshot, will evolve
        W_4 = dict(self.layers[4].named_parameters())   # snapshot, will evolve
        W_0_3 = {k: v.clone() for k, v in W_3.items()}  # base for anchor
        W_0_4 = {k: v.clone() for k, v in W_4.items()}  # base for anchor
        s = torch.zeros(B, 64, device=x.device)          # GRU navigation state
        
        # Previous chunk logits (for computing errors)
        prev_logits = torch.zeros(B, 1, self.vocab_size, device=x.device)
        # prev_errors: [B, 1, d_err] — errors for the last token of previous chunk
        prev_e = torch.zeros(B, 64, device=x.device)
        
        all_logits = []
        
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        
        for c in range(num_chunks):
            chunk_start = c * self.chunk_size
            chunk_end   = min(chunk_start + self.chunk_size, T)
            chunk_len   = chunk_end - chunk_start
            
            x_chunk = x[:, chunk_start:chunk_end]  # [B, chunk_len, d]
            ids_chunk = input_ids[:, chunk_start:chunk_end]
            
            # ============================================================
            # STEP 1: Apply multiplicative update BEFORE processing chunk
            # Based on navigation state from previous chunk
            # ============================================================
            u3, v3, gate3, u4, v4, gate4 = self.meta_net.get_update(s)
            W_3 = self._apply_update(W_3, u3, v3, gate3, W_0_3)
            W_4 = self._apply_update(W_4, u4, v4, gate4, W_0_4)
            
            # ============================================================
            # STEP 2: Forward pass through model IN PARALLEL for this chunk
            # W_3 and W_4 are fixed during this chunk
            # ============================================================
            h = x_chunk
            
            # Lower layers (fixed, fully parallel)
            for i in range(3):  # layers 0, 1, 2
                h = self.layers[i](h)
            h_lower = h.clone()  # save for GRU input
            
            # SURGE layer 3 (with current W_3)
            h = self._forward_with_weights(self.layers[3], h, W_3)
            
            # SURGE layer 4 (with current W_4)
            h = self._forward_with_weights(self.layers[4], h, W_4)
            
            # Upper layers (fixed, fully parallel)
            for i in range(5, 9):  # layers 5, 6, 7, 8
                h = self.layers[i](h)
            
            # Output logits for this chunk
            chunk_logits = self.output_head(h)  # [B, chunk_len, vocab]
            all_logits.append(chunk_logits)
            
            # ============================================================
            # STEP 3: Compute prediction errors for THIS chunk
            # (Used to update GRU state for NEXT chunk's update)
            # error_t = one_hot(token_t) - softmax(logits_{t-1})
            # For t=0 in chunk, use last logit from previous chunk
            # ============================================================
            # Shift logits: logit for predicting token t is logits[t-1]
            # For first token of chunk, use last logit from prev chunk
            if c == 0:
                shifted_logits = torch.cat([
                    torch.zeros(B, 1, self.vocab_size, device=x.device),
                    chunk_logits[:, :-1]
                ], dim=1)  # [B, chunk_len, vocab]
            else:
                last_prev_logit = prev_logits[:, -1:]  # [B, 1, vocab]
                shifted_logits  = torch.cat([
                    last_prev_logit,
                    chunk_logits[:, :-1]
                ], dim=1)  # [B, chunk_len, vocab]
            
            # Compute errors for each token in chunk
            errors_chunk = []
            for t_local in range(chunk_len):
                e_t = self.meta_net.compress_error(
                    shifted_logits[:, t_local],   # logits_{t-1}
                    ids_chunk[:, t_local]          # actual token_t
                )  # [B, d_err]
                errors_chunk.append(e_t)
            
            # ============================================================
            # STEP 4: Update GRU navigation state (sequential but cheap)
            # GRU integrates token representations + prediction errors
            # ============================================================
            for t_local in range(chunk_len):
                h_t = h_lower[:, t_local]      # [B, d_model] — lower layer repr
                e_t = errors_chunk[t_local]    # [B, d_err]
                s   = self.meta_net.update_state(h_t, e_t, s)
            
            # Save for next chunk
            prev_logits = chunk_logits
        
        logits = torch.cat(all_logits, dim=1)  # [B, T, vocab]
        return logits
    
    def _apply_update(self, W_dict, u, v, gate, W_0_dict, max_drift=0.1):
        """Apply multiplicative update to W_O only."""
        W_new = {}
        for k, W in W_dict.items():
            if 'attn.W_O' in k or 'c_proj' in k or 'out_proj' in k:
                # This is the output projection — apply multiplicative update
                W_new[k] = apply_multiplicative_update(W, u, v, gate, W_0_dict[k], max_drift)
            else:
                # Other weights: no update
                W_new[k] = W
        return W_new
    
    def _forward_with_weights(self, layer, h, W_dict):
        """
        Run a transformer layer forward pass with EXPLICIT weights
        instead of the stored nn.Parameters.
        
        This requires the transformer block to accept explicit weight arguments,
        or use functional APIs.
        
        Implementation note: easiest approach is to temporarily swap the layer's
        parameters with W_dict values using a context manager.
        """
        # See implementation section for details
        pass
```

---

## 7. Critical Implementation Detail: Running Layer with Modified Weights

The most technically tricky part is running a transformer layer with modified W_O while W_O's original value is stored as an nn.Parameter.

**Recommended approach: Functional forward**

Instead of modifying the parameters in-place, implement the transformer block to accept optional explicit weight overrides:

```python
class SURGETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp  = MLP(config)
        self.ln1  = nn.LayerNorm(config.d_model)
        self.ln2  = nn.LayerNorm(config.d_model)
    
    def forward(self, x, W_O_override=None):
        """
        Standard transformer block forward.
        If W_O_override is provided, use it instead of self.attn.W_O.
        """
        # Self-attention
        residual = x
        x = self.ln1(x)
        attn_out = self.attn(x, W_O_override=W_O_override)
        x = residual + attn_out
        
        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)  # This is what evolves
        self.n_heads = config.n_heads
    
    def forward(self, x, W_O_override=None):
        B, T, d = x.shape
        
        # Standard QKV computation (using stored weights)
        Q = self.W_Q(x)  # [B, T, d]
        K = self.W_K(x)
        V = self.W_V(x)
        
        # ... reshape, compute attention ...
        attn_out = # [B, T, d] result of attention
        
        # Output projection: use override if provided
        if W_O_override is not None:
            # W_O_override is [d, d], apply manually
            out = F.linear(attn_out.reshape(B*T, d), W_O_override)
            out = out.reshape(B, T, d)
        else:
            out = self.W_O(attn_out)
        
        return out
```

**Alternative: Parameter swap approach**

```python
import contextlib

@contextlib.contextmanager
def temp_param_override(module, overrides):
    """
    Context manager to temporarily override specific parameters.
    Restores original values on exit.
    """
    originals = {}
    for name, value in overrides.items():
        originals[name] = module.get_parameter(name).data.clone()
        module.get_parameter(name).data.copy_(value)
    try:
        yield
    finally:
        for name, value in originals.items():
            module.get_parameter(name).data.copy_(value)
```

**Recommendation**: Use the explicit W_O_override argument approach. It's cleaner and avoids in-place operations that could cause issues with autograd.

---

## 8. Training Procedure

### Loss

Standard next-token prediction loss:
```python
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    input_ids[:, 1:].reshape(-1),
    ignore_index=-1
)
```

### Gradient Flow

The computation graph looks like:

```
Sequence of W_t values:
W_0 → [chunk 1 output: u1, v1, g1] → W_1 → [chunk 2 output: u2, v2, g2] → W_2 → ...

Loss at token t in chunk c depends on:
  - W_c (via SURGE layer forward pass)
  - W_c depends on (u_c, v_c, g_c) from M
  - (u_c, v_c, g_c) depend on s_c (GRU state at end of chunk c-1)
  - s_c depends on (h_lower, errors) from all tokens 0...(c*64-1)
  - errors depend on logits from previous chunk which depend on W_{c-1}
```

**FOMAML approximation**: when computing gradient through W_c → W_{c-1}:
- `W_c = W_{c-1} + delta` where `W_{c-1}` is detached
- This means: gradient of loss flows INTO (u_c, v_c, g_c) → M's output heads → M's GRU state
- BUT does NOT flow through W_{c-1} back into earlier chunks' (u,v,g) values

This is the first-order approximation. Without it, you'd need second-order gradients (expensive).

### Implementation: Truncated BPTT

```python
def train_step(model, input_ids, optimizer, tbptt_chunks=4):
    """
    Train one sequence with truncated BPTT through the last K chunks.
    """
    B, T = input_ids.shape
    chunk_size = model.chunk_size
    num_chunks = T // chunk_size
    
    # Process first (num_chunks - tbptt_chunks) without gradient
    with torch.no_grad():
        W_3, W_4, s = model.forward_chunks_no_grad(
            input_ids[:, :chunk_size * (num_chunks - tbptt_chunks)]
        )
    
    # Process last tbptt_chunks with gradient
    W_3 = {k: v.detach().requires_grad_(False) for k, v in W_3.items()}
    W_4 = {k: v.detach().requires_grad_(False) for k, v in W_4.items()}
    s   = s.detach()  # detach GRU state from earlier history
    
    logits_grad, _ = model.forward_chunks_with_grad(
        input_ids[:, chunk_size * (num_chunks - tbptt_chunks):],
        W_3_init=W_3, W_4_init=W_4, s_init=s
    )
    
    target = input_ids[:, chunk_size * (num_chunks - tbptt_chunks) + 1:]
    loss = F.cross_entropy(
        logits_grad[:, :-1].reshape(-1, model.vocab_size),
        target.reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()
```

**Alternative (simpler)**: Since FOMAML detaches W updates, gradients don't flow across chunk boundaries through W anyway. The GRU recurrence gradient is the main concern. With truncated GRU gradient over K=4 chunks, you can just run the full forward pass and let autograd handle it — only 4*64=256 tokens of GRU gradient history.

```python
# Simpler approach: just run full forward pass with gradient
# The FOMAML detach in apply_multiplicative_update handles the W chain
# The GRU recurrence only needs 4 chunks of gradient (256 tokens)
# This is manageable for T=1024

logits = model(input_ids)
loss = CE(logits[:, :-1], input_ids[:, 1:])
loss.backward()  # autograd handles the truncated graph correctly
```

### Optimizers

```python
# Separate parameter groups
base_params   = [p for n, p in model.named_parameters() if 'meta_net' not in n]
meta_params   = list(model.meta_net.parameters())

optimizer_base = MuonOptimizer(
    [{'params': [p for p in base_params if p.dim() >= 2], 'lr': 1e-3},
     {'params': [p for p in base_params if p.dim() < 2],  'lr': 1e-3}]
)

optimizer_meta = torch.optim.AdamW(
    meta_params,
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Step both optimizers together
def optimizer_step():
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_base.step()
    optimizer_meta.step()
    optimizer_base.zero_grad()
    optimizer_meta.zero_grad()
```

---

## 9. Hyperparameters

```python
# === Architecture (SAME as competition baseline) ===
VOCAB_SIZE    = 1024   # SentencePiece
D_MODEL       = 256    # residual stream width
N_HEADS       = 4      # attention heads
N_KV_HEADS    = 4      # KV heads (GQA - same as baseline)
N_LAYERS      = 9      # total transformer blocks
D_FF          = 512    # MLP expansion (2x)
MAX_SEQ_LEN   = 1024

# === SURGE specifics ===
SURGE_LAYERS  = [3, 4] # which layers have evolving W_O
CHUNK_SIZE    = 64     # tokens per chunk (update frequency)
UPDATE_RANK   = 1      # rank of u ⊗ v update

# === Meta-network M ===
D_ERR         = 64     # error compression dimension
D_STATE       = 64     # GRU hidden state

# === Stability ===
MAX_DRIFT_FRACTION = 0.1  # max ||W_t - W_0|| / ||W_0|| (Frobenius norm)
GATE_INIT_BIAS     = -4.6  # sigmoid(-4.6) ≈ 0.01 → near-zero initial updates

# === Training ===
TBPTT_CHUNKS  = 4      # truncate BPTT through this many chunks (256 tokens)
LR_BASE       = 1e-3   # Muon learning rate
LR_META       = 3e-4   # AdamW learning rate for M
GRAD_CLIP     = 1.0    # max gradient norm
BATCH_SIZE    = 32     # same as baseline
GRAD_ACCUM    = 8      # same as baseline
WALLCLOCK_CAP = 600    # 10 minutes
```

---

## 10. What to Measure (Logging)

Beyond standard BPB logging, track:

```python
def log_surge_diagnostics(model, step):
    """Track SURGE-specific metrics to understand what M is learning."""
    
    # 1. How much have weights drifted from base?
    drift_3 = (model.W_3_current - model.W_0_3).norm(p='fro').item()
    drift_4 = (model.W_4_current - model.W_0_4).norm(p='fro').item()
    norm_3  = model.W_0_3.norm(p='fro').item()
    norm_4  = model.W_0_4.norm(p='fro').item()
    
    # 2. What are M's gate values? (Are updates actually happening?)
    # Sample a batch and compute average gate values
    mean_gate_3 = # avg over batch and sequence positions
    mean_gate_4 = # avg over batch and sequence positions
    
    # 3. What's the magnitude of u and v?
    # (Zero M outputs → standard transformer)
    u_norm_3 = # mean norm of u vectors output by M
    v_norm_3 = # mean norm of v vectors
    
    # 4. Error signal magnitude (is surprisal being computed?)
    mean_error_magnitude = # mean ||e_t|| across sequence
    
    log({
        'drift_layer3': drift_3 / norm_3,
        'drift_layer4': drift_4 / norm_4,
        'gate_3_mean': mean_gate_3,
        'gate_4_mean': mean_gate_4,
        'u_norm_3': u_norm_3,
        'v_norm_3': v_norm_3,
        'error_magnitude': mean_error_magnitude,
        'bpb': current_bpb,
    })
```

**Expected early training behavior**:
- Drift ~= 0 (gates near zero, M not producing updates yet)
- BPB ≈ baseline (model is effectively standard transformer)

**Expected late training behavior**:
- Drift > 0 (M producing non-zero updates for surprising tokens)
- Gate values vary across sequences (different domains trigger different updates)
- BPB: unclear — could be better or same as baseline

---

## 11. Failure Mode Handling

### M never learns (stays at zero)
**Symptom**: gate values stay near 0.01 throughout training
**Cause**: gradient into M's output heads is too small / meta_params LR too low
**Fix**: 
1. Increase LR_META to 1e-3
2. Try GATE_INIT_BIAS = -2.3 (sigmoid gives 0.09 → larger initial updates)
3. Add auxiliary loss: encourage M to produce non-zero updates

### Catastrophic drift
**Symptom**: BPB spikes, drift fraction > 0.5
**Cause**: gate value too large, u/v vectors too large
**Fix**:
1. Reduce MAX_DRIFT_FRACTION to 0.05
2. Add L2 regularization on u and v output: loss += 0.001 * (u.norm() + v.norm())
3. Reduce LR_META

### Gradient explosion
**Symptom**: loss NaN or very large, gradients explode in GRU
**Cause**: GRU recurrence over 4 chunks (256 tokens) with large state changes
**Fix**:
1. Reduce TBPTT_CHUNKS to 2
2. Reduce D_STATE to 32
3. Use gradient clipping per-module, not just globally

### Too slow (< 800 steps in 10 min)
**Symptom**: ms/step > 800ms
**Cause**: sequential GRU processing too slow
**Fix**:
1. Reduce D_STATE to 32 (faster GRU)
2. Increase CHUNK_SIZE to 128 (fewer chunk boundaries, less GRU overhead)
3. Reduce D_ERR to 32

---

## 12. Comparison Experiments to Run

### Experiment A: SURGE-M (main experiment)
Full architecture as described. Config above.

### Experiment B: Ablation — Additive update
Same as SURGE-M but change:
```python
W_new = W_prev + gate_mean * torch.outer(u_mean, v_mean)  # additive, not multiplicative
```
Compare BPB to Experiment A. If A > B: multiplicative matters.

### Experiment C: Ablation — Scalar surprisal only
Same as SURGE-M but compress error to 1 dimension (surprisal magnitude) instead of 64:
```python
self.err_proj = nn.Sequential(nn.Linear(vocab_size, 1))
# GRU input: [h_lower (d_model), surprisal (1)]
```
Compare BPB. If A > C: directional error matters.

### Experiment D: Ablation — No GRU (memoryless M)
Same as SURGE-M but M is a 2-layer MLP with no hidden state:
```python
(u, v, gate) = MLP(cat(mean_h_chunk, mean_e_chunk))
```
Compare BPB. If A > D: navigation history (GRU state) matters.

Run experiments B, C, D only if time permits after A. They answer the key scientific questions.

---

## 13. Connection to Existing Work

### How SURGE-M differs from Titans (NeurIPS 2025)

Titans maintains a SEPARATE neural memory module M_t. The main processing layers are frozen. M_t stores KV associations, updated by gradient of associative memory loss. Updates are additive (gradient descent).

SURGE-M updates the PROCESSING weights W_t of the main transformer. The meta-network M is a GRU that navigates, not computes gradients. Updates are multiplicative (function composition). The model literally BECOMES a different function at each chunk, not merely having access to an additional memory.

### How SURGE-M differs from TTT-E2E

TTT-E2E uses the next-token prediction gradient to update MLP weights (additive). No meta-network, no navigation state, no directional error signal.

SURGE-M: (1) multiplicative updates, (2) recurrent M maintains navigation history, (3) prediction error vector not just loss value.

### Relationship to MAML

SURGE-M resembles MAML with:
- Inner loop: M updates W_t for each chunk
- Outer loop: standard next-token prediction trains both W_0 and θ_M
- FOMAML approximation: W_t detached from W_{t-1} chain

But SURGE-M is trained end-to-end as a sequence model, not as a meta-learner over tasks.

---

## 14. File Structure

```
surge_m/
├── README.md                  ← this file
├── model.py                   ← SURGE_M_Model, MetaNetwork, SURGEBlock
├── attention.py               ← MultiHeadAttention with W_O_override
├── train.py                   ← training loop, optimizer setup
├── utils.py                   ← apply_multiplicative_update, logging
└── experiments/
    ├── exp_A_surge_m.py       ← main experiment
    ├── exp_B_additive.py      ← ablation: additive updates
    ├── exp_C_scalar.py        ← ablation: scalar surprisal
    └── exp_D_memoryless.py    ← ablation: no GRU
```

---

## 15. The Pitch

> Standard language models process every token with the same frozen function. TTT and Titans add external memory. We do neither — we change the function itself.
>
> At each chunk of 64 tokens, a recurrent meta-network reads directional prediction errors and navigates in the space of programs: it outputs a rank-1 multiplicative update that potentially changes what computation the model performs. Not nudging parameters — composing new functions.
>
> A multiplicative update (I + gate * u ⊗ v) can change the eigenspectrum of the weight matrix. For a network run repeatedly (like a Universal Transformer), this changes the attractor — the fixed point of computation. A small multiplicative perturbation can cause the model to think differently, not just slightly differently.
>
> The meta-network is trained to learn when to make such transitions and in which direction. Its GRU remembers the trajectory through program space — past updates, past errors, what kind of text has been flowing. It learns the grammar of program transitions.
>
> We are not building a better memory. We are building a model that learns how to change itself.

---

## Appendix: Quick Reference

```python
# The core update equation
delta = gate_mean * torch.outer(u_mean, v_W)  # v_W = v^T @ W_prev
W_new = W_prev + delta                          # efficient implementation of
                                                # (I + gate * u⊗v) @ W_prev
# Stability
if (W_new - W_0).norm() > 0.1 * W_0.norm():
    W_new = W_0 + (W_new - W_0) * (0.1 * W_0.norm() / (W_new - W_0).norm())

# What M sees at each token
gru_input = cat([lower_layers_output, compressed_prediction_error])

# What M outputs at each chunk boundary
u, v = M.heads(gru_state)   # rank-1 update factors
gate = sigmoid(M.gate(gru_state))  # [0, 1] scalar

# Why this isn't just SGD
# SGD: W_new = W_old - η * ∇L  (additive, local step in tangent space)
# SURGE: W_new = (I + delta) @ W_old (multiplicative, function composition)
# These are equivalent only for infinitesimal delta.
# For finite delta, multiplicative changes eigenspectrum; additive does not.
```
