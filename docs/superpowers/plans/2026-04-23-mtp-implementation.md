# Multi-Token Prediction (MTP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 2 auxiliary prediction heads (+2, +3 tokens) during training to improve representation quality, with lambda annealing to zero so aux params never enter the artifact.

**Architecture:** Two `Linear(512,512)` transforms after final norm, each projecting through shared `tok_emb.weight` to predict future tokens. Combined aux loss weighted by λ=0.3, annealed to 0 over last 30% of training. Training-only — excluded from serialization.

**Tech Stack:** PyTorch, single-file modification to `train_gpt.py`

---

### Task 1: Add MTP hyperparameters

**Files:**
- Modify: `train_gpt.py:39-88` (Hyperparameters class)

- [ ] **Step 1: Add MTP env vars to Hyperparameters**

Add these lines after line 88 (`grad_clip_norm`):

```python
    # Multi-Token Prediction
    mtp_enabled = bool(int(os.environ.get("MTP_ENABLED", "0")))
    mtp_lambda = float(os.environ.get("MTP_LAMBDA", 0.3))
    mtp_anneal_start = float(os.environ.get("MTP_ANNEAL_START", 0.7))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 2))
```

- [ ] **Step 2: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add MTP hyperparameters"
```

---

### Task 2: Add MTPHead module and integrate into GPT

**Files:**
- Modify: `train_gpt.py:606-724` (after MLP class, modify GPT class)

- [ ] **Step 1: Add MTPHead class after the MLP class (after line 618)**

```python
class MTPHead(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.transform = CastedLinear(model_dim, model_dim, bias=False)
        self.transform._zero_init = True

    def forward(self, hidden: Tensor, tok_emb_weight: Tensor, softcap: float) -> Tensor:
        h = self.transform(hidden) + hidden
        logits = F.linear(h, tok_emb_weight)
        return softcap * torch.tanh(logits / softcap)
```

Note: Uses `CastedLinear` (existing class at line 509) to match the codebase pattern. `_zero_init = True` is handled by the existing `_init_weights` loop at line 696-698.

- [ ] **Step 2: Add mtp_heads to GPT.__init__**

In `GPT.__init__`, add a `mtp_num_heads` parameter and create the heads. After line 690 (`self.lm_head._zero_init = True`), before `self._init_weights()`:

```python
        self.mtp_heads = nn.ModuleList(
            [MTPHead(model_dim) for _ in range(mtp_num_heads)]
        ) if mtp_num_heads > 0 else nn.ModuleList()
```

Update the `__init__` signature to accept `mtp_num_heads: int = 0`.

- [ ] **Step 3: Modify GPT.forward to support MTP**

Replace lines 700-724 with:

```python
    def forward(self, input_ids: Tensor, target_ids: Tensor, mtp_lambda: float = 0.0) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        if mtp_lambda > 0.0 and self.training and len(self.mtp_heads) > 0:
            aux_loss = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)
            for k, head in enumerate(self.mtp_heads):
                shift = k + 2
                h = x[:, :-shift, :].reshape(-1, x.size(-1))
                t = target_ids[:, shift:].reshape(-1)
                aux_logits = head(h, self.tok_emb.weight, self.logit_softcap)
                aux_loss = aux_loss + F.cross_entropy(aux_logits.float(), t, reduction="mean")
            return main_loss + mtp_lambda * aux_loss / len(self.mtp_heads)

        return main_loss
```

- [ ] **Step 4: Commit**

```bash
git add train_gpt.py
git commit -m "feat: add MTPHead module and integrate into GPT forward"
```

---

### Task 3: Wire MTP into training loop and optimizer

**Files:**
- Modify: `train_gpt.py:826-838` (model construction)
- Modify: `train_gpt.py:846-893` (optimizer setup)
- Modify: `train_gpt.py:1007-1018` (training loop forward call)

- [ ] **Step 1: Pass mtp_num_heads to GPT constructor**

At line 826, add `mtp_num_heads` to the GPT call:

```python
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads if args.mtp_enabled else 0,
    ).to(device).bfloat16()
```

- [ ] **Step 2: Add MTP head params to scalar optimizer**

After line 863 (`scalar_params.append(base_model.skip_weights)`), add:

```python
    for head in base_model.mtp_heads:
        for p in head.parameters():
            scalar_params.append(p)
```

This puts MTP head params in the Adam optimizer with `scalar_lr`. Since they use `CastedLinear` (2D), they'd normally go into Muon via `block_named_params`, but MTP heads aren't in `base_model.blocks` — they're at the top level. The `block_named_params` filter on line 851 only captures `base_model.blocks.named_parameters()`, so MTP params are already excluded from Muon. We just need to make sure they're picked up by some optimizer.

Actually, MTP heads are `nn.ModuleList` at the GPT level, not inside `blocks`. The current code only collects params from `base_model.blocks` (line 851) and `base_model.skip_weights` (line 862-863). MTP params would be orphaned. We need to add them explicitly.

- [ ] **Step 3: Compute mtp_lambda schedule and pass to forward**

In the training loop, after line 1008 (`scale = lr_mul(step, elapsed_ms)`), add the lambda computation:

```python
        mtp_lambda = 0.0
        if args.mtp_enabled:
            total_steps_approx = max(int(args.max_wallclock_seconds * 1000.0 / max(elapsed_ms / max(step, 1), 1.0)), step + 1)
            progress = step / total_steps_approx
            if progress >= args.mtp_anneal_start:
                mtp_lambda = args.mtp_lambda * (1.0 - (progress - args.mtp_anneal_start) / (1.0 - args.mtp_anneal_start))
            else:
                mtp_lambda = args.mtp_lambda
```

Then modify the forward call at line 1016:

```python
                loss = model(x, y, mtp_lambda=mtp_lambda)
```

Also update the warmup forward call at line 948:

```python
                    warmup_loss = model(x, y, mtp_lambda=args.mtp_lambda if args.mtp_enabled else 0.0)
```

- [ ] **Step 4: Log MTP config**

After line 910 (`log0(f"seed:{args.seed}")`), add:

```python
    if args.mtp_enabled:
        mtp_params = sum(p.numel() for h in base_model.mtp_heads for p in h.parameters())
        log0(f"mtp:enabled heads:{args.mtp_num_heads} lambda:{args.mtp_lambda} anneal_start:{args.mtp_anneal_start} params:{mtp_params}")
```

- [ ] **Step 5: Commit**

```bash
git add train_gpt.py
git commit -m "feat: wire MTP into training loop, optimizer, and lambda schedule"
```

---

### Task 4: Exclude MTP heads from serialization

**Files:**
- Modify: `train_gpt.py:1068-1069` (model save)
- Modify: `train_gpt.py:1076` (quantization)

- [ ] **Step 1: Filter MTP params from saved state dict**

At line 1069, replace:
```python
        torch.save(base_model.state_dict(), "final_model.pt")
```
with:
```python
        save_state = {k: v for k, v in base_model.state_dict().items() if not k.startswith("mtp_")}
        torch.save(save_state, "final_model.pt")
```

At line 1076, replace:
```python
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
```
with:
```python
    quant_state = {k: v for k, v in base_model.state_dict().items() if not k.startswith("mtp_")}
    quant_obj, quant_stats = quantize_state_dict_int8(quant_state)
```

- [ ] **Step 2: Handle roundtrip load with strict=False for MTP params**

At line 1099, the roundtrip validation loads the quantized state dict back. Since MTP params were excluded, we need `strict=False` or to only load matching keys:

Replace:
```python
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
```
with:
```python
    roundtrip_state = dequantize_state_dict_int8(quant_state)
    base_model.load_state_dict(roundtrip_state, strict=False)
```

This is safe because eval mode doesn't use MTP heads (they're gated by `self.training` and `mtp_lambda > 0.0`).

- [ ] **Step 3: Commit**

```bash
git add train_gpt.py
git commit -m "feat: exclude MTP heads from serialization and quantization"
```

---

### Task 5: Local smoke test

**Files:** None (verification only)

- [ ] **Step 1: Quick CPU smoke test**

Run a minimal sanity check that the code parses and the model can be constructed with MTP enabled:

```bash
python3 -c "
import os
os.environ['MTP_ENABLED'] = '1'
os.environ['MTP_NUM_HEADS'] = '2'
os.environ['VOCAB_SIZE'] = '1024'
os.environ['NUM_LAYERS'] = '4'
os.environ['MODEL_DIM'] = '128'
os.environ['NUM_HEADS'] = '4'
os.environ['NUM_KV_HEADS'] = '2'
os.environ['MLP_MULT'] = '2'

# Import just the classes
import importlib.util, sys
spec = importlib.util.spec_from_file_location('tgp', 'train_gpt.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['tgp'] = mod

import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

# We need to exec just the class definitions
exec(open('train_gpt.py').read().split('def main')[0])

model = GPT(
    vocab_size=1024, num_layers=4, model_dim=128,
    num_heads=4, num_kv_heads=2, mlp_mult=2,
    tie_embeddings=True, tied_embed_init_std=0.005,
    logit_softcap=30.0, rope_base=10000.0,
    qk_gain_init=1.5, mtp_num_heads=2,
)
x = torch.randint(0, 1024, (1, 32))
y = torch.randint(0, 1024, (1, 32))

# Test without MTP
loss_no_mtp = model(x, y, mtp_lambda=0.0)
print(f'Loss without MTP: {loss_no_mtp.item():.4f}')

# Test with MTP
model.train()
loss_with_mtp = model(x, y, mtp_lambda=0.3)
print(f'Loss with MTP (lambda=0.3): {loss_with_mtp.item():.4f}')

# Verify MTP adds to loss
assert loss_with_mtp.item() >= loss_no_mtp.item() - 0.1, 'MTP loss should be >= main loss'

# Verify MTP params excluded from filtered state dict
full_keys = set(model.state_dict().keys())
filtered_keys = {k for k in full_keys if not k.startswith('mtp_')}
mtp_keys = full_keys - filtered_keys
print(f'Total params keys: {len(full_keys)}, MTP keys: {len(mtp_keys)}, Saved keys: {len(filtered_keys)}')
assert len(mtp_keys) > 0, 'Should have MTP keys'
assert all(k.startswith('mtp_') for k in mtp_keys)

print('All smoke tests passed!')
"
```

Expected: All assertions pass, prints loss values and key counts.

- [ ] **Step 2: Commit (if any fixes needed)**

```bash
git add train_gpt.py
git commit -m "fix: smoke test corrections for MTP"
```

---

### Task 6: GPU integration test

**Files:** None (verification on RunPod)

- [ ] **Step 1: Run a short MTP training test on GPU**

On the RunPod server, run a quick 100-step test to verify MTP works end-to-end with distributed training, compilation, and serialization:

```bash
MTP_ENABLED=1 MTP_LAMBDA=0.3 MTP_NUM_HEADS=2 \
ITERATIONS=100 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=120 \
SEED=42 RUN_ID=mtp_smoke \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Verify in the log:
1. `mtp:enabled heads:2 lambda:0.3 anneal_start:0.7 params:...` appears
2. Training runs without errors
3. Serialization completes (no MTP keys in saved model)
4. Roundtrip validation produces a valid val_bpb

- [ ] **Step 2: Compare step time with and without MTP**

Run the same 100 steps without MTP for timing comparison:

```bash
MTP_ENABLED=0 ITERATIONS=100 VAL_LOSS_EVERY=50 MAX_WALLCLOCK_SECONDS=120 \
SEED=42 RUN_ID=baseline_smoke \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Compare `step_avg` between the two runs. MTP overhead should be <5ms.

- [ ] **Step 3: Commit any fixes**

---

### Task 7: Full training run and evaluation

**Files:** None (execution on RunPod)

- [ ] **Step 1: Launch full 600s MTP training run**

```bash
MTP_ENABLED=1 MTP_LAMBDA=0.3 MTP_NUM_HEADS=2 MTP_ANNEAL_START=0.7 \
SEED=1337 RUN_ID=mtp_full_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

- [ ] **Step 2: Compare val_bpb against baseline (1.0783)**

If val_bpb improves by >=0.003 (target: <=1.0753), proceed to 3-seed runs. If not, tune MTP_LAMBDA and MTP_ANNEAL_START.

- [ ] **Step 3: Record results**

Create `records/track_10min_16mb/2026-04-23_MTP_AuxHeads/` with README.md, logs, and train_gpt.py.
