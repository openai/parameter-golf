# R2-15: Adaptive Causal Probing — Revised Design

## Idea

For tokens the model already predicts correctly, apply aggressive context dropout to stress-test which context is truly causally necessary. Inverse of Rho-1: instead of skipping easy tokens, make them harder.

## Design Revision: Per-Batch Adaptive Rate

### Problem with original spec
Token dropout removes entire columns from the sequence — you can't drop context for easy positions only, because all positions share the same context. The dropout mask is applied to the whole sequence.

### Corrected approach: Adaptive dropout rate per batch
Instead of per-position dropout, adapt the **dropout rate** based on how well the model is doing:

```
Step 1: Forward pass on current batch → per-position loss L_i
Step 2: Compute easy_fraction = fraction of positions where argmax(logits) == target
Step 3: Set next batch's dropout rate proportional to easy_fraction:
        drop_rate = base_rate + (aggressive_rate - base_rate) × easy_fraction
        
When model predicts 80% correctly → drop_rate ≈ 0.26 (aggressive)
When model predicts 30% correctly → drop_rate ≈ 0.13 (gentle)
When model predicts 5% correctly  → drop_rate ≈ 0.06 (minimal)
```

**Parameters**:
- `base_rate = 0.05` (minimum dropout, even when struggling)
- `aggressive_rate = 0.30` (maximum dropout, when model is confident)

This creates an automatic curriculum:
- **Early training** (model predicts ~5%): near-zero dropout, clean data to learn basics
- **Mid training** (model predicts ~50%): moderate dropout, building robustness
- **Late training** (model predicts ~80%): aggressive dropout, learning causal dependencies

### Why this is better than uniform corruption (R2-11)
R2-11 corrupted context uses a fixed 10% rate throughout training. Adaptive probing:
- Starts gentler (doesn't disrupt early learning)
- Gets more aggressive as the model improves (challenges confident predictions)
- Is informed by the model's actual performance, not a fixed schedule

## Mathematical Spec

```
State: prev_easy_frac = 0.0  (initialized to 0 — no dropout on first step)

For each training step:
  1. Compute adaptive dropout rate:
     drop_rate = base_rate + (aggressive_rate - base_rate) × prev_easy_frac
  
  2. Apply token dropout at this rate:
     mask = Bernoulli(1 - drop_rate) for each position
     mask[0] = True  (keep first token)
     x = x[:, mask]
     y = y[:, mask]
  
  3. Forward pass → get loss
     To get per-position info, we need reduction="none":
     per_pos_loss = model(x, y, reduction="none")     # [B*T']
     loss = per_pos_loss.mean()                         # scalar for backward
  
  4. Update easy fraction for next step:
     with torch.no_grad():
       # Reconstruct logits from the forward pass (need model change)
       # OR: use loss threshold as proxy
       easy_frac = (per_pos_loss < easy_threshold).float().mean()
       prev_easy_frac = 0.9 * prev_easy_frac + 0.1 * easy_frac  # EMA smoothing
```

### EMA smoothing
The easy fraction is smoothed with exponential moving average (α=0.1) to avoid oscillation:
- If one batch is unusually easy → don't spike dropout to 30% instantly
- If one batch is unusually hard → don't collapse dropout to 0% instantly
- The EMA provides a stable, gradually adapting difficulty signal

### Easy threshold
Use `easy_threshold = 1.0` (loss < 1.0 nat ≈ model assigns >37% probability to correct token). This is a reasonable proxy for "the model roughly knows this token."

## Interface

### Env vars
- `CAUSAL_PROBE=1` — enable adaptive causal probing
- `CAUSAL_PROBE_BASE=0.05` — minimum dropout rate
- `CAUSAL_PROBE_MAX=0.30` — maximum dropout rate
- `CAUSAL_PROBE_THRESHOLD=1.0` — loss threshold for "easy" classification
- `CAUSAL_PROBE_EMA=0.1` — EMA smoothing factor

### Code changes required

#### 1. GPT.forward() — add `reduction` parameter
```python
def forward(self, input_ids, target_ids, reduction="mean"):
    ...
    logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction=reduction)
```
One line changed, fully backward compatible.

#### 2. Hyperparameters class — add env vars
```python
causal_probe = bool(int(os.environ.get("CAUSAL_PROBE", "0")))
causal_probe_base = float(os.environ.get("CAUSAL_PROBE_BASE", 0.05))
causal_probe_max = float(os.environ.get("CAUSAL_PROBE_MAX", 0.30))
causal_probe_threshold = float(os.environ.get("CAUSAL_PROBE_THRESHOLD", 1.0))
causal_probe_ema = float(os.environ.get("CAUSAL_PROBE_EMA", 0.1))
```

#### 3. Training loop — adaptive dropout logic
```python
# Before training loop
prev_easy_frac = 0.0

# Inside micro_step loop, REPLACE the existing token dropout block:
if args.causal_probe and base_model.training:
    # Adaptive rate based on previous batch's difficulty
    drop_rate = args.causal_probe_base + (args.causal_probe_max - args.causal_probe_base) * prev_easy_frac
    if drop_rate > 0:
        mask = torch.rand(x.shape[1], device=x.device) > drop_rate
        mask[0] = True
        x = x[:, mask]
        y = y[:, mask]

# Forward pass with per-position loss
if args.causal_probe and base_model.training:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        per_pos_loss = model(x, y, reduction="none")
        loss = per_pos_loss.mean()
    # Update easy fraction (detached, no gradient)
    with torch.no_grad():
        easy_frac = (per_pos_loss.detach() < args.causal_probe_threshold).float().mean().item()
        prev_easy_frac = (1 - args.causal_probe_ema) * prev_easy_frac + args.causal_probe_ema * easy_frac
else:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss = model(x, y)
```

### Lines of code: ~20 (plus 5 env vars)

## Parameter Budget
Zero — data augmentation only.

## Verifiable DoD
1. `model(x, y, reduction="none")` returns shape `[B*T]` (per-position loss)
2. `model(x, y, reduction="mean")` returns scalar (backward compatible)
3. `model(x, y)` returns scalar (default behavior unchanged)
4. At `CAUSAL_PROBE=0`: no behavior change, standard training
5. At step 0: `prev_easy_frac=0` → `drop_rate=base_rate=0.05` (gentle start)
6. When model predicts 80% correctly: `drop_rate ≈ 0.25` (aggressive)
7. EMA is detached — no gradient flows through difficulty estimation
8. Dropout rate is logged for monitoring: `log0(f"causal_probe drop_rate:{drop_rate:.3f} easy_frac:{prev_easy_frac:.3f}")`

## Interaction with existing features
- **Replaces** R2-5 token dropout and R2-11 corrupted context (subsumes both)
- **Compatible with** all MLP variants, BigramHash, XSA, value residual
- **Incompatible with** R2-8 graduated dropout and R2-12 graduated corruption (different scheduling approach)

## Expected Outcome
- Should outperform uniform corruption (R2-11, val_bpb=1.3004) because dropout rate adapts to model's actual performance
- The automatic curriculum means no manual schedule tuning needed
- Risk: EMA may be too slow to track rapid learning phase transitions

## Experiment Configs
```bash
# Adaptive causal probing (recommended)
CAUSAL_PROBE=1 CAUSAL_PROBE_BASE=0.05 CAUSAL_PROBE_MAX=0.30

# Ablation: compare with uniform corruption at same average rate
CORRUPT_RATE=0.15  # approximate midpoint of adaptive range

# Ablation: adaptive + DML-Gated (combine best architecture + best augmentation)
CAUSAL_PROBE=1 MLP_TYPE=dml_gated BT_LAMBDA=0.01
```
