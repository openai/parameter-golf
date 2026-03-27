# LoRA Stability Fix Plan

The LoRA per-pass adapters are causing training instability (40× growth ratios, loss spiking to 28.9). Three root causes, all in `train_gpt_recurrent.py`. Apply all fixes.

---

## Fix 1: Add rsLoRA scaling to the forward pass

**File:** `train_gpt_recurrent.py`
**Location:** `GPT._forward_hidden`, inside the core loop where LoRA is applied

The raw `B @ A` product is added to weights with no scaling. At rank 8, the output magnitude is √8 ≈ 2.83× too large. Apply `α/√r` scaling (rsLoRA).

**Find this block:**
```python
if self.lora_rank > 0:
    ci = j - self.core_start
    q_w   = q_w   + self.lora_B_q[k, ci]   @ self.lora_A_q[k, ci]
    k_w   = k_w   + self.lora_B_k[k, ci]   @ self.lora_A_k[k, ci]
    v_w   = v_w   + self.lora_B_v[k, ci]   @ self.lora_A_v[k, ci]
    out_w = out_w + self.lora_B_out[k, ci]  @ self.lora_A_out[k, ci]
    up_w  = up_w  + self.lora_B_up[k, ci]   @ self.lora_A_up[k, ci]
    down_w = down_w + self.lora_B_down[k, ci] @ self.lora_A_down[k, ci]
```

**Replace with:**
```python
if self.lora_rank > 0:
    ci = j - self.core_start
    s = self._lora_scale  # precomputed 1.0 / sqrt(rank)
    q_w   = q_w   + s * (self.lora_B_q[k, ci]   @ self.lora_A_q[k, ci])
    k_w   = k_w   + s * (self.lora_B_k[k, ci]   @ self.lora_A_k[k, ci])
    v_w   = v_w   + s * (self.lora_B_v[k, ci]   @ self.lora_A_v[k, ci])
    out_w = out_w + s * (self.lora_B_out[k, ci]  @ self.lora_A_out[k, ci])
    up_w  = up_w  + s * (self.lora_B_up[k, ci]   @ self.lora_A_up[k, ci])
    down_w = down_w + s * (self.lora_B_down[k, ci] @ self.lora_A_down[k, ci])
```

**Also add in `GPT.__init__`**, after the LoRA parameter creation block:
```python
self._lora_scale = 1.0 / math.sqrt(lora_rank) if lora_rank > 0 else 1.0
```

---

## Fix 2: Change LoRA initialization from kaiming to zero

**File:** `train_gpt_recurrent.py`
**Location:** `GPT.__init__`, the LoRA parameter creation block

Kaiming init on A makes `||A||` ≈ √dim ≈ 22.6. After one gradient step on B, `||BA||` is proportional to this — far too large. Both A and B should start at zero so the LoRA is a no-op at initialization and learns its contribution gradually.

**Find this block:**
```python
if lora_rank > 0 and self.num_core > 0 and num_passes > 1:
    nc, np_, r = self.num_core, num_passes, lora_rank
    for wname, in_d, out_d in [
        ("q", model_dim, model_dim), ("out", model_dim, model_dim),
        ("k", model_dim, kv_dim),    ("v", model_dim, kv_dim),
        ("up", model_dim, mlp_dim),  ("down", mlp_dim, model_dim),
    ]:
        A = nn.Parameter(torch.empty(np_, nc, r, in_d))
        B = nn.Parameter(torch.zeros(np_, nc, out_d, r))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        setattr(self, f"lora_A_{wname}", A)
        setattr(self, f"lora_B_{wname}", B)
```

**Replace with:**
```python
if lora_rank > 0 and self.num_core > 0 and num_passes > 1:
    nc, np_, r = self.num_core, num_passes, lora_rank
    for wname, in_d, out_d in [
        ("q", model_dim, model_dim), ("out", model_dim, model_dim),
        ("k", model_dim, kv_dim),    ("v", model_dim, kv_dim),
        ("up", model_dim, mlp_dim),  ("down", mlp_dim, model_dim),
    ]:
        A = nn.Parameter(torch.zeros(np_, nc, r, in_d))
        B = nn.Parameter(torch.zeros(np_, nc, out_d, r))
        setattr(self, f"lora_A_{wname}", A)
        setattr(self, f"lora_B_{wname}", B)
    self._lora_scale = 1.0 / math.sqrt(lora_rank)
```

---

## Fix 3: Give LoRA params their own optimizer group with lower learning rate

**File:** `train_gpt_recurrent.py`
**Location:** `main()`, in the optimizer setup section

Currently LoRA params are added to `extra_scalar_params` and trained at `scalar_lr=0.025`. LoRA matrices are 2D weight matrices, not scalars — they need a separate, lower learning rate.

**Find this block:**
```python
if base_model.lora_rank > 0:
    lora_params = [p for n, p in base_model.named_parameters() if "lora_" in n]
    for p in lora_params:
        p.data = p.data.float()
    extra_scalar_params.extend(lora_params)
    log0(f"lora: rank={base_model.lora_rank} params={sum(p.numel() for p in lora_params)}")
```

**Replace with:**
```python
if base_model.lora_rank > 0:
    lora_params = [p for n, p in base_model.named_parameters() if "lora_" in n]
    for p in lora_params:
        p.data = p.data.float()
    # Do NOT add to extra_scalar_params — LoRA gets its own optimizer
    log0(f"lora: rank={base_model.lora_rank} params={sum(p.numel() for p in lora_params)}")
```

**Then, after `optimizer_scalar` is created, add a new optimizer:**
```python
optimizer_lora = None
if base_model.lora_rank > 0:
    lora_lr = args.scalar_lr * 0.1  # 10× lower than scalar_lr
    optimizer_lora = torch.optim.AdamW(
        [{"params": lora_params, "lr": lora_lr, "base_lr": lora_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    # Add LoRA params to replicated_params for distributed all-reduce
    replicated_params.extend(lora_params)
    log0(f"lora_optimizer: lr={lora_lr} (scalar_lr * 0.1)")
```

**Update the optimizers list** (find where it's defined):
```python
optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
if optimizer_head is not None:
    optimizers.append(optimizer_head)
if optimizer_lora is not None:
    optimizers.append(optimizer_lora)
```

---

## Fix 4: Reduce default LoRA rank from 8 to 2

**File:** `train_gpt_recurrent.py`
**Location:** `Hyperparameters` class and CLI args

Rank 8 across 6 weight types × 5 core layers × 4 passes = 1.2M params of perturbation surface. Rank 2 gives 307K params — enough for per-pass differentiation, small enough that the Jacobian proxy loss can control it.

**In `Hyperparameters`:**
```python
lora_rank = int(os.environ.get("LORA_RANK", 0))  # no change needed, default is already 0
```

**In the run script, change `--lora-rank 8` to `--lora-rank 2`:**
```bash
--lora-rank 2
```

---

## Fix 5: Increase Jacobian proxy weight for LoRA runs

With LoRA perturbations, the Jacobian proxy loss needs to work harder to keep things contractive. The per-pass weight deltas create additional expansive directions that the loss must counteract.

**In the run script:**
```bash
--jacobian-proxy-weight 0.1
```

This was already discussed but confirm it's set to 0.1, not 0.01.

---

## Summary

| Fix | What | Where | Impact |
|-----|------|-------|--------|
| 1 | rsLoRA scaling `1/√r` | `_forward_hidden` + `__init__` | Reduces LoRA output magnitude by √r |
| 2 | Zero init A and B | `__init__` LoRA creation | LoRA is no-op at init, learns gradually |
| 3 | Separate optimizer at 0.1× LR | `main()` optimizer setup | Prevents LoRA params from overshooting |
| 4 | Rank 8 → rank 2 | Run script CLI arg | 4× less perturbation surface |
| 5 | Jacobian weight 0.01 → 0.1 | Run script CLI arg | Stronger contractivity pressure |

## Test command after fixes:
```bash
NUM_PASSES=4 \
CORE_START=3 \
CORE_END=8 \
ITERATIONS=500 \
VAL_LOSS_EVERY=50 \
TRAIN_LOG_EVERY=10 \
python train_gpt_recurrent.py \
    --feedback-mode diagonal \
    --feedback-rank 2 \
    --jacobian-proxy-weight 0.1 \
    --lora-rank 2 \
    --no-interpass-rmsnorm
```

## Expected behavior after fixes:
- Growth ratios at step 0: ~1.0-1.2 (same as without LoRA, since LoRA is zero-initialized)
- Growth ratios at step 50: ~1.0-1.3 (LoRA starting to contribute, Jacobian loss keeping it in check)
- No loss spikes above 10 in the first 20 steps
- Train loss should track the non-LoRA 4-pass run closely for the first ~100 steps, then gradually improve as LoRA learns per-pass specialization
