# Recurrent SOTA: Complete Fix Plan

All fixes listed below are required. Apply them in order.

---

## Fix 1: Add inter-pass RMSNorm (CRITICAL)

The core layers were trained to expect inputs at a specific scale. On pass 2+, the output of the core has a different magnitude than the input. Without renormalization, pass 2 feeds out-of-distribution activations into the same weights, causing 35-48× growth.

**In `GPT.forward()` and `GPT.forward_logits()`, in the core loop:**

```python
# --- RECURRENT CORE ---
for k in range(self.num_passes):
    if k > 0:
        x = F.rms_norm(x, (x.size(-1),))
    for j in range(self.core_start, self.core_end):
        # ... layer execution
```

Zero extra parameters. This is what Universal Transformers and Huginn both do.

---

## Fix 2: Move feedback call outside the inner layer loop

The feedback module is designed to correct at **junction points between passes**, not at every layer within a pass. Currently `feedback_fn(x, k)` is called inside the `for j` loop, meaning it fires 5 times per pass (once per core layer) instead of once per pass.

**Before (wrong):**
```python
for k in range(self.num_passes):
    for j in range(self.core_start, self.core_end):
        correction = feedback_fn(x, k) if feedback_fn else None
        if correction is not None:
            x = x + correction
        # ... layer execution
```

**After (correct):**
```python
for k in range(self.num_passes):
    if k > 0:
        x = F.rms_norm(x, (x.size(-1),))
    # Junction correction: once per pass, before re-entering core
    if feedback_fn is not None:
        correction = feedback_fn(x, k)
        if correction is not None:
            x = x + correction
    if stabilizer is not None:
        x = stabilizer.clip(x)
    h_core_in = x  # save for Jacobian proxy loss
    for j in range(self.core_start, self.core_end):
        h_prev = x
        ve = self._get_ve(j, input_ids, ve_cache)
        q_w, k_w, v_w, out_w, up_w, down_w = self._get_bank_weights(j)
        x, raw_v = self.blocks[j](x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
            v_embed=ve, v0=v0)
        if v0 is None and raw_v is not None:
            v0 = raw_v
        if stabilizer is not None and self.training and not torch.compiler.is_compiling():
            stabilizer.record_pass(h_prev, x)
    h_core_out = x  # save for Jacobian proxy loss
```

---

## Fix 3: Zero-initialize the feedback module

The `LowRankResidual` U/V matrices are initialized with random values, and `DiagonalFeedback.d` is initialized to ones. At step 1, this injects random noise at the same magnitude as the hidden state. The feedback module must be a no-op at initialization so it can't hurt before it's learned anything.

**In `feedback.py`, change `LowRankResidual.__init__`:**
```python
class LowRankResidual(nn.Module):
    def __init__(self, dim: int, rank: int = 2):
        super().__init__()
        self.V = nn.Parameter(torch.zeros(dim, rank))
        self.U = nn.Parameter(torch.zeros(dim, rank))
```

**In `feedback.py`, change `DiagonalFeedback.__init__` default:**
```python
class DiagonalFeedback(nn.Module):
    def __init__(self, dim: int, init_ones: bool = False):  # was True
        super().__init__()
        init_val = torch.ones(dim) if init_ones else torch.zeros(dim)
        self.d = nn.Parameter(init_val)
```

**In `feedback.py`, change `LowRankFeedback.__init__`:**
```python
class LowRankFeedback(nn.Module):
    def __init__(self, dim: int, rank: int = 2):
        super().__init__()
        self.V_D = nn.Parameter(torch.zeros(dim, rank))
        self.U_D = nn.Parameter(torch.zeros(dim, rank))
```

---

## Fix 4: Wire up Jacobian proxy loss in the training loop

The Jacobian proxy loss penalizes the spectral norm of the core block's Jacobian exceeding 1.0. This is the training-time mechanism that ensures the recurrent core is **contractive** — meaning quantization errors shrink rather than grow across passes. Without it, the model has no incentive to learn a stable recurrence.

**Step A: Have forward() return Jacobian proxy inputs.**

Change the forward signature to optionally return core boundary activations:

```python
def forward(self, input_ids, target_ids, feedback_fn=None, stabilizer=None,
            return_jacobian_pair=False):
    # ... stem ...
    
    h_core_in = None
    h_core_out = None
    
    # --- RECURRENT CORE ---
    for k in range(self.num_passes):
        if k > 0:
            x = F.rms_norm(x, (x.size(-1),))
        if feedback_fn is not None:
            correction = feedback_fn(x, k)
            if correction is not None:
                x = x + correction
        if stabilizer is not None:
            x = stabilizer.clip(x)
        if k == 0:
            h_core_in = x
        for j in range(self.core_start, self.core_end):
            # ... layer execution ...
            pass
        if k == self.num_passes - 1:
            h_core_out = x
    
    # ... tail + loss computation ...
    
    if return_jacobian_pair and h_core_in is not None and h_core_out is not None:
        return main_loss, h_core_in, h_core_out
    return main_loss
```

**Step B: Add Jacobian loss in the training loop in `main()`:**

```python
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch(...)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        if stabilizer is not None and stabilizer.jacobian_proxy_weight > 0:
            loss, h_in, h_out = model(x, y, feedback_fn=feedback_fn,
                                       stabilizer=stabilizer,
                                       return_jacobian_pair=True)
            loss = loss + stabilizer.jacobian_proxy_loss(h_in, h_out)
        else:
            loss = model(x, y, feedback_fn=feedback_fn, stabilizer=stabilizer)
    train_loss += loss.detach()
    (loss * grad_scale).backward()
```

**Step C: Set a non-zero default weight.** Change the CLI default:
```python
g.add_argument("--jacobian-proxy-weight", type=float, default=0.01)
```

Start with 0.01. If training is too slow to converge (the regularizer fights the language modeling loss), reduce to 0.001. If growth ratios are still above 1.5, increase to 0.1.

---

## Fix 5: Wire up ResidualScale in the forward pass

`ResidualScale` is instantiated but never called. It dampens the residual update on each pass, giving the model a learnable per-pass attenuation factor.

**In `GPT.__init__`, add `residual_scale` as a constructor arg:**
```python
def __init__(self, ..., residual_scale: nn.Module | None = None):
    # ...
    self.residual_scale = residual_scale
```

**In the core loop, apply it to the block's residual output. This requires changing how Block output is used:**

The cleanest approach — apply ResidualScale at the pass level, scaling the entire pass's contribution to x:

```python
for k in range(self.num_passes):
    if k > 0:
        x = F.rms_norm(x, (x.size(-1),))
    if feedback_fn is not None:
        correction = feedback_fn(x, k)
        if correction is not None:
            x = x + correction
    if stabilizer is not None:
        x = stabilizer.clip(x)
    
    x_before_pass = x
    for j in range(self.core_start, self.core_end):
        # ... layer execution, x gets updated ...
        pass
    
    # Scale the residual delta of this pass
    if self.residual_scale is not None and k > 0:
        delta = x - x_before_pass
        x = x_before_pass + self.residual_scale(delta, k)
```

Pass `residual_scale` through from `main()`:
```python
base_model = GPT(..., residual_scale=residual_scale)
```

Initialize `residual_scale` with init_value=0.5 (start conservative, let it learn up):
```python
g.add_argument("--residual-scale-init", type=float, default=0.5)
```

---

## Fix 6: Factor out `_forward_hidden` to eliminate duplication

`forward()` and `forward_logits()` duplicate the entire stem/core/tail logic. Every fix above must be applied to both, which is a maintenance nightmare and a guaranteed source of bugs.

**Create a shared method:**
```python
def _forward_hidden(self, input_ids, feedback_fn=None, stabilizer=None,
                    return_jacobian_pair=False):
    """Run stem/core/tail, return (hidden_states, v0, jacobian_pair_or_None)."""
    n = self.num_layers
    x = self.tok_emb(input_ids)
    if self.bigram is not None:
        x = x + self.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = self.smear(x)
    x0 = x
    v0 = None
    skips = []
    ve_cache = {}
    
    # --- STEM ---
    for i in range(self.core_start):
        ve = self._get_ve(i, input_ids, ve_cache)
        q_w, k_w, v_w, out_w, up_w, down_w = self._get_bank_weights(i)
        x, raw_v = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
            v_embed=ve, v0=v0)
        if v0 is None and raw_v is not None:
            v0 = raw_v
        skips.append(x)
    
    # --- RECURRENT CORE ---
    h_core_in = None
    h_core_out = None
    for k in range(self.num_passes):
        if k > 0:
            x = F.rms_norm(x, (x.size(-1),))
        if feedback_fn is not None:
            correction = feedback_fn(x, k)
            if correction is not None:
                x = x + correction
        if stabilizer is not None:
            x = stabilizer.clip(x)
        if k == 0:
            h_core_in = x
        x_before_pass = x
        for j in range(self.core_start, self.core_end):
            h_prev = x
            ve = self._get_ve(j, input_ids, ve_cache)
            q_w, k_w, v_w, out_w, up_w, down_w = self._get_bank_weights(j)
            x, raw_v = self.blocks[j](x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            if stabilizer is not None and self.training and not torch.compiler.is_compiling():
                stabilizer.record_pass(h_prev, x)
        if self.residual_scale is not None and k > 0:
            delta = x - x_before_pass
            x = x_before_pass + self.residual_scale(delta, k)
        if k == self.num_passes - 1:
            h_core_out = x
    
    # --- TAIL ---
    for i in range(self.core_end, n):
        ti = i - self.core_end
        if ti < len(skips):
            x = x + self.skip_weights[ti].to(dtype=x.dtype)[None, None, :] * skips.pop()
        ve = self._get_ve(i, input_ids, ve_cache)
        q_w, k_w, v_w, out_w, up_w, down_w = self._get_bank_weights(i)
        x, _ = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
            v_embed=ve, v0=v0)
    
    x = self.final_norm(x)
    
    jac_pair = (h_core_in, h_core_out) if return_jacobian_pair and h_core_in is not None else None
    return x, jac_pair
```

**Then forward() and forward_logits() become thin wrappers:**
```python
def forward(self, input_ids, target_ids, feedback_fn=None, stabilizer=None,
            return_jacobian_pair=False):
    x, jac_pair = self._forward_hidden(input_ids, feedback_fn, stabilizer,
                                        return_jacobian_pair)
    x_flat = x.reshape(-1, x.size(-1))
    targets = target_ids.reshape(-1)
    if self.tie_embeddings:
        logits_proj = F.linear(x_flat, self.tok_emb.weight)
    else:
        logits_proj = self.lm_head(x_flat)
    logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
    # ... MTP loss computation stays here ...
    if jac_pair is not None:
        return main_loss, jac_pair[0], jac_pair[1]
    return main_loss

def forward_logits(self, input_ids, feedback_fn=None, stabilizer=None):
    x, _ = self._forward_hidden(input_ids, feedback_fn, stabilizer, False)
    if self.tie_embeddings:
        logits_proj = F.linear(x, self.tok_emb.weight)
    else:
        logits_proj = self.lm_head(x)
    return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
```

---

## Fix 7: torch.compile compatibility for Jacobian loss

The `return_jacobian_pair=True` path returns a tuple instead of a single tensor. `torch.compile(fullgraph=True)` will break if the return type changes dynamically. Two options:

**Option A (simpler):** Always return the tuple, ignore jac_pair when not needed:
```python
# Always return 3 values
def forward(self, ...):
    # ...
    return main_loss, h_core_in, h_core_out  # h_core_in/out can be None
```

**Option B (safer for compile):** Compute Jacobian loss inside forward() so the compiled function always returns a scalar:
```python
def forward(self, input_ids, target_ids, feedback_fn=None, stabilizer=None):
    x, jac_pair = self._forward_hidden(input_ids, feedback_fn, stabilizer, True)
    # ... compute main_loss ...
    if jac_pair is not None and stabilizer is not None and stabilizer.jacobian_proxy_weight > 0:
        main_loss = main_loss + stabilizer.jacobian_proxy_loss(jac_pair[0], jac_pair[1])
    return main_loss
```

**Go with Option B.** It keeps the compiled function signature stable and avoids any compile issues. Pass the stabilizer through — the Jacobian loss computation is pure tensor ops and compile-friendly.

---

## Summary: Apply in this order

| # | Fix | Files changed | Risk |
|---|-----|--------------|------|
| 1 | Inter-pass RMSNorm | train_gpt_recurrent.py | None — proven technique |
| 2 | Move feedback outside inner loop | train_gpt_recurrent.py | None — bug fix |
| 3 | Zero-initialize feedback | feedback.py | None — strictly safer init |
| 4 | Wire Jacobian proxy loss | train_gpt_recurrent.py, stability.py | Low — use Option B for compile safety |
| 5 | Wire ResidualScale | train_gpt_recurrent.py | Low — init at 0.5 is conservative |
| 6 | Factor out _forward_hidden | train_gpt_recurrent.py | None — refactor only |
| 7 | torch.compile compatibility | train_gpt_recurrent.py | None — use Option B |

After applying all fixes, the recommended first run config:
```bash
NUM_PASSES=2 \
CORE_START=3 \
CORE_END=8 \
torchrun --standalone --nproc_per_node=8 train_gpt_recurrent.py \
    --feedback-mode diagonal \
    --feedback-rank 2 \
    --clip-hidden \
    --clip-value 15 \
    --residual-scale-init 0.5 \
    --jacobian-proxy-weight 0.01
```

Expected behavior: growth ratios should be 0.8-1.2 per layer (stable), val_bpb should converge to competitive range within the first 1000 steps, and the inter-pass RMSNorm alone should prevent the 35-48× explosions seen earlier.
