# Novel Code Snippets — Parameter Golf

Techniques from recent PRs that we haven't integrated yet.

---

## 1. Full-Weight SGD TTT (PR #264, #281)

**Replace LoRA TTT with full-model SGD on val data:**

```python
# After training, before final eval
if args.ttt_enabled:
    log0("TTT: Full-weight SGD on val data")
    ttt_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.ttt_lr,  # 0.002
        momentum=args.ttt_momentum,  # 0.9
    )
    
    for epoch in range(args.ttt_epochs):  # 2-3 epochs
        for batch in val_loader:
            x, y = batch
            loss = model(x, y)
            loss.backward()
            ttt_optimizer.step()
            ttt_optimizer.zero_grad()
    
    log0(f"TTT: Adapted {args.ttt_epochs} epochs on val")
```

**Impact:** ~0.005 BPB gain vs LoRA  
**Risk:** Low  

---

## 2. LAWA-EMA (Continuous EMA, not periodic SWA)

**Replace periodic SWA checkpointing with every-step EMA:**

```python
# In training loop setup
lawa_ema_state = None
lawa_decay = float(os.environ.get("LAWA_DECAY", 0.995))

# In training loop (every step during warmdown)
if scale < args.swa_start_frac:
    if lawa_ema_state is None:
        lawa_ema_state = copy.deepcopy(base_model.state_dict())
        log0(f"LAWA-EMA: started with decay={lawa_decay}")
    else:
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                lawa_ema_state[name] = (
                    lawa_decay * lawa_ema_state[name] 
                    + (1 - lawa_decay) * param.detach().cpu()
                )

# After training
if lawa_ema_state is not None:
    log0("LAWA-EMA: Applying continuous average")
    base_model.load_state_dict(
        {k: v.to(device=device) for k, v in lawa_ema_state.items()},
        strict=True
    )
```

**Impact:** ~0.001-0.002 BPB gain vs periodic SWA  
**Risk:** Low  
**Source:** PR #machdragon, #0xjaishy mentions

---

## 3. Force FlashAttention 2 (not cuDNN SDPA)

**PR #281 finding: cuDNN is 40% faster but worse BPB**

```python
# Early in main(), before model creation
import torch.backends.cuda

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

log0("Forced FlashAttention 2 SDPA backend (disabled cuDNN)")
```

**Impact:** ~0.004 BPB if currently using cuDNN  
**Risk:** Very low (just a flag)

---

## 4. Attention Sigmoid Gate (PR #mattqlf)

**Add learnable gate after attention output:**

```python
# In CausalSelfAttention.__init__()
self.attn_gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

# In CausalSelfAttention.forward(), after computing attn_out
gate = torch.sigmoid(self.attn_gate.to(dtype=attn_out.dtype))
attn_out = attn_out * gate[None, None, :]  # elementwise gating
```

**Impact:** ~0.001-0.003 BPB (speculative, only 1 mention)  
**Risk:** Very low (3 lines)  
**Source:** PR #mattqlf commit message

---

## 5. Int5 MLP + Int6 Attention (PR #264)

**Use 5-bit quantization for MLP, 6-bit for attention:**

```python
def quantize_int5_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize to [-16, 15] (5 bits) instead of [-32, 31]"""
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 15.0).clamp_min(1e-12).to(torch.float16)  # Changed: 31 → 15
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -16, 15).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 15.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -16, 15).to(torch.int8)
    return q, scale

def mixed_quantize_int5_int6(state_dict: dict[str, Tensor]):
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        cat = _classify_param(name)
        if cat == "mlp":
            q, s = quantize_int5_per_row(tensor)  # 5-bit for MLP
            result[name] = q
            meta[name + ".scale"] = s
        elif cat in ("attn", "embed"):
            q, s = quantize_int6_per_row(tensor)  # 6-bit for attn/embed
            result[name] = q
            meta[name + ".scale"] = s
        else:
            result[name] = tensor  # control tensors stay fp16
    return result, meta
```

**Impact:** Saves ~1.9MB vs uniform Int6 → can fund 12th layer or wider model  
**Risk:** Medium (need to verify zstd compression ratio)  
**Source:** PR #264

---

## 6. Weight Decay as Artifact Size Controller

**Systematic sweep to target 15.5MB:**

```python
# Hyperparameters
muon_wd = float(os.environ.get("MUON_WD", 0.042))  # Was 0.04
adam_wd = float(os.environ.get("ADAM_WD", 0.042))  # Was 0.04

# Results from PR #281:
# WD=0.040 → 16.3MB (invalid)
# WD=0.041 → 15.6MB, 1.1378 BPB
# WD=0.042 → 15.5MB, 1.1374 BPB ✅ OPTIMAL
# WD=0.045 → 15.6MB, 1.1466 BPB (over-regularized)
# WD=0.050 → 15.0MB, 1.1418 BPB (too small)
```

**Impact:** ~0.003 BPB + ensures artifact stays under 16MB  
**Risk:** Very low (just tune one hyperparameter)  
**Source:** PR #236, PR #281

---

## 7. RoPE Base 50K (for seq2048)

**Better position interpolation at long context:**

```python
rope_base = float(os.environ.get("ROPE_BASE", 50000.0))  # Was 10000.0
```

**Impact:** ~0.001 BPB for seq2048 training  
**Risk:** Very low  
**Source:** PR #0xjaishy, #michaeljabbour

---

## 8. Smaller Batch Tokens (More Gradient Updates)

**524288 vs 786432 → more steps per wallclock second:**

```python
train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524288))  # Was 786432
```

**Impact:** ~0.002 BPB (more gradient updates in fixed time)  
**Risk:** Very low  
**Source:** PR #236, PR #281

---

## 9. SWA Every 200 Steps (not 50)

**Fewer, later checkpoints:**

```python
swa_every = int(os.environ.get("SWA_EVERY", 200))  # Was 50
swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))  # Start at warmdown
```

**Impact:** ~0.001 BPB (less noisy averaging)  
**Risk:** Very low  
**Source:** PR #281

---

## 10. Code Minification (Budget Hack)

**Shrink code from ~69KB → ~40KB to free budget for model params:**

```python
# Techniques from PR #281 "What We'd Try Next":
# 1. Remove comments
# 2. Shorten variable names (hyperparameters → hp, model → m)
# 3. Remove docstrings
# 4. Compress imports
# 5. Use shorter control flow (ternary, list comprehensions)

# Example:
# Before:
class Hyperparameters:
    """Training hyperparameters loaded from environment variables."""
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    
# After:
class HP:
    v,l=int(os.environ.get("V",1024)),int(os.environ.get("L",9))
```

**Impact:** Frees ~29KB → +2.4M params → ~0.003-0.005 BPB  
**Risk:** Medium (harder to debug)  
**Source:** PR #281 recommendations

---

## 11. MoE with Learned Token Routing (PR #250)

**Advanced, high-complexity:**

```python
class TokenRoutedMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, mlp_mult: float = 3.0):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            MLP(dim, mlp_mult) for _ in range(num_experts)
        ])
        self.router = nn.Linear(dim, num_experts)  # Learned routing
        
    def forward(self, x: Tensor) -> Tensor:
        # Soft routing (differentiable)
        router_logits = self.router(x)  # [B, T, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, E]
        
        # Weighted sum over experts
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            out = out + router_probs[..., i:i+1] * expert_out
        
        return out
```

**Impact:** ~0.01 BPB if tuned correctly (huge upside)  
**Risk:** Very high (unstable training, needs CUDA kernels for efficiency)  
**Source:** PR #250

---

## Summary of Quick Wins

**Lowest effort, highest confidence:**
1. Force FlashAttention 2 (1 line)
2. WD=0.042 (1 line)
3. RoPE base 50K (1 line)
4. Smaller batch tokens (1 line)
5. SWA every 200 steps (1 line)

**Medium effort, high upside:**
6. Full-weight SGD TTT (replace LoRA logic, ~20 lines)
7. LAWA-EMA (replace SWA logic, ~30 lines)
8. Attention sigmoid gate (3 lines)

**High effort, moonshot:**
9. Int5 MLP + Int6 attn (modify quantization, ~50 lines)
10. Code minification (architectural refactor)
11. MoE with learned routing (full rewrite, ~200+ lines)
