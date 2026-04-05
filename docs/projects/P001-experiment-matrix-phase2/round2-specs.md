# Round 2: Formal Specifications

## R2-1: FAN Periodic MLP

### Mathematical Spec
Replace standard MLP activation with FAN (Fourier Analysis Network) layer:
```
Standard MLP:   output = down_proj(leaky_relu(up_proj(x), 0.5)²)

FAN MLP:        p = W_p · x                                    (shared Fourier projection)
                fan_out = [cos(p) ∥ sin(p) ∥ leaky_relu(W_bar · x, 0.5)²]   (concatenate)
                output = W_out · fan_out
```
Where `∥` denotes concatenation along the feature dimension.

**Dimension split** (at hidden_dim=1536 for 3x MLP):
- `d_p = hidden_dim × 0.25 = 384` → produces 384 cos + 384 sin = 768 Fourier dims
- `d_bar = hidden_dim - 2×d_p = 1536 - 768 = 768` → standard activation dims
- Total output: 768 + 768 = 1536 (same as standard MLP)

### Interface
- **New class**: `FAN_MLP(dim=512, hidden_dim=1536, fourier_ratio=0.25)`
- **Replaces**: Standard MLP class
- **Env var**: `MLP_TYPE=fan` (default `standard`)

### Parameter Budget
| Component | Standard MLP | FAN MLP |
|-----------|-------------|---------|
| `up_proj` [512, 1536] | 786,432 | — |
| `W_p` [512, 384] | — | 196,608 |
| `W_bar` [512, 768] | — | 393,216 |
| `down_proj` / `W_out` [1536, 512] | 786,432 | 786,432 |
| **Total per layer** | **1,572,864** | **1,376,256** |
| **Delta** | — | **-196,608 (-12.5%)** |

FAN is actually **cheaper** than standard MLP — `W_p` is shared for both sin and cos, saving one projection matrix.

### Initialization
- `W_p`: orthogonal, gain=1.0
- `W_bar`: orthogonal, gain=1.0
- `W_out`: zero-init, scaled by `1/sqrt(2*num_layers)`

### Verifiable DoD
1. `assert model.blocks[0].mlp.W_p.weight.shape == (384, 512)`
2. `assert model.blocks[0].mlp.W_bar.weight.shape == (768, 512)`
3. `assert model.blocks[0].mlp.W_out.weight.shape == (512, 1536)`
4. **Fourier output test**: Given `x = torch.randn(1, 10, 512)`:
   ```python
   p = mlp.W_p(x)                  # [1, 10, 384]
   cos_p = torch.cos(p)            # [1, 10, 384]
   sin_p = torch.sin(p)            # [1, 10, 384]
   assert cos_p.shape == sin_p.shape == (1, 10, 384)
   assert torch.allclose(cos_p**2 + sin_p**2, torch.ones_like(cos_p))  # trig identity
   ```
5. Full forward: `mlp(x).shape == (1, 10, 512)` — preserves model dim
6. Param count per layer < standard MLP param count

---

## R2-2: DML-Gated MLP + Barlow Twins Loss

### Mathematical Spec
```
Forward:
  h_gate = leaky_relu(W_nuisance · x, 0.5)²        ("nuisance" pathway — gates)
  h_value = W_target · x                              ("target" pathway — values)
  output = W_out · (h_gate ⊙ h_value)                 (element-wise gating)

Auxiliary loss (training only):
  z_n = normalize(W_nuisance · x, μ=0, σ=1)          (batch-standardize nuisance activations)
  z_t = normalize(W_target · x, μ=0, σ=1)            (batch-standardize target activations)
  C = (z_n^T · z_t) / N_batch                         (cross-correlation matrix [hidden, hidden])
  L_BT = Σ_{i≠j} C²_{ij}                             (minimize off-diagonal = decorrelation)
  L_total = L_CE + λ · L_BT                           (λ = 0.01 default)
```

### Interface
- **New class**: `DML_GatedMLP(dim=512, hidden_dim=1024)`
  - Note: hidden_dim = 2/3 × 1536 = 1024 for 3-matrix budget neutrality
- **New method**: `barlow_twins_loss(x) → scalar`
- **Training loop change**: Add `λ * sum(mlp.barlow_twins_loss(x) for mlp in model.mlps)` to total loss
- **Env var**: `MLP_TYPE=dml_gated`, `BT_LAMBDA=0.01`

### Parameter Budget
| Component | Params |
|-----------|--------|
| `W_nuisance` [512, 1024] | 524,288 |
| `W_target` [512, 1024] | 524,288 |
| `W_out` [1024, 512] | 524,288 |
| **Total per layer** | **1,572,864** |

Exactly matches standard 3x MLP (1536×512×2 = 1,572,864). Budget-neutral.

### Initialization
- `W_nuisance`: orthogonal, gain=1.0
- `W_target`: orthogonal, gain=1.0
- `W_out`: zero-init, scaled by `1/sqrt(2*num_layers)`

### Barlow Twins Loss Details
- Computed on pre-activation values (before leaky_relu/square)
- Batch standardization: subtract mean, divide by std, per-feature across batch×seq dims
- Cross-correlation C is `[hidden_dim, hidden_dim]` — could be expensive at hidden=1024
- **Optimization**: Sample subset of features (e.g., 128 random dims) to reduce cost from O(1024²) to O(128²)
- λ = 0.01 (start small, tune if needed)

### Verifiable DoD
1. `assert model.blocks[0].mlp.W_nuisance.weight.shape == (1024, 512)`
2. `assert model.blocks[0].mlp.W_target.weight.shape == (1024, 512)`
3. **Gating test**: `gate * value` has same shape as each pathway output
4. **BT loss test**: 
   ```python
   bt_loss = mlp.barlow_twins_loss(x)
   assert bt_loss.shape == ()  # scalar
   assert bt_loss >= 0         # always non-negative
   ```
5. **Decorrelation check at init**: With orthogonal init, `bt_loss` should be near-zero initially
6. **Gradient flows** through both pathways and through BT loss

---

## R2-3: DML with Gram-Schmidt Orthogonalization

### Mathematical Spec
```
Forward:
  h_n = leaky_relu(W_nuisance · x, 0.5)²            ("nuisance" features)
  h_t = leaky_relu(W_target · x, 0.5)²               ("target" features)
  ĥ_n = h_n / ||h_n||₂                               (normalize nuisance, dim=-1)
  proj = (h_t · ĥ_n) · ĥ_n                            (project target onto nuisance direction)
  h_t_orth = h_t - proj                                (remove nuisance component from target)
  output = W_out · [h_n ∥ h_t_orth]                    (concatenate and project)
```

### Interface
- **New class**: `DML_OrthMLP(dim=512, hidden_dim=768)`
  - Nuisance: `[512, 384]` (half width)
  - Target: `[512, 768]` (full width)
  - Combiner: `[384+768, 512] = [1152, 512]`
- **No auxiliary loss** — orthogonality enforced structurally in forward pass
- **Env var**: `MLP_TYPE=dml_orth`

### Parameter Budget
| Component | Params |
|-----------|--------|
| `W_nuisance` [512, 384] | 196,608 |
| `W_target` [512, 768] | 393,216 |
| `W_out` [1152, 512] | 589,824 |
| **Total per layer** | **1,179,648** |
| **vs standard** | **-393,216 (-25%)** |

Under budget — could increase target width to compensate.

### Numerical Stability
- Add epsilon to normalization: `ĥ_n = h_n / (||h_n||₂ + 1e-8)`
- The projection is differentiable — gradients flow through both `h_n` and `h_t`

### Verifiable DoD
1. **Orthogonality test** (the key invariant):
   ```python
   h_n, h_t_orth = mlp.forward_with_components(x)
   h_n_norm = F.normalize(h_n, dim=-1)
   dot = (h_t_orth * h_n_norm).sum(dim=-1)
   assert dot.abs().max() < 1e-5  # orthogonal
   ```
2. Forward output shape: `mlp(x).shape == (B, T, 512)`
3. Gradient flows through both pathways
4. When `h_n ≈ 0` (edge case): no NaN/inf from normalization

---

## R2-4: FAN + DML-Gated Combo

### Mathematical Spec
```
Forward:
  p = W_p · x                                         (Fourier projection)
  fan_features = [cos(p) ∥ sin(p)]                     (periodic features)
  std_features = leaky_relu(W_bar · x, 0.5)²          (standard features)
  
  gate = leaky_relu(W_gate · x, 0.5)²                 (DML gate)
  value = [fan_features ∥ std_features]                 (concatenate)
  output = W_out · (gate ⊙ value)                      (gated combination)
  
  + Barlow Twins loss between gate and value activations
```

### Interface
- **New class**: `FAN_DML_MLP(dim=512, hidden_dim=1024, fourier_ratio=0.25)`
- **Env var**: `MLP_TYPE=fan_dml`

### Parameter Budget
| Component | Params |
|-----------|--------|
| `W_p` [512, 256] | 131,072 |
| `W_bar` [512, 512] | 262,144 |
| `W_gate` [512, 1024] | 524,288 |
| `W_out` [1024, 512] | 524,288 |
| **Total per layer** | **1,441,792** |
| **vs standard** | **-131,072 (-8.3%)** |

### Verifiable DoD
1. Forward output shape correct
2. Fourier features satisfy trig identity: `cos²(p) + sin²(p) = 1`
3. Barlow Twins loss is scalar, non-negative
4. Gradient flows through all four pathways

---

## R2-5: Token Dropout (10%)

### Mathematical Spec
```
During training only:
  mask_i ~ Bernoulli(1 - drop_rate)     for each position i
  mask[0] = 1                            (always keep first token)
  input_ids' = input_ids[mask]           (compact)
  target_ids' = target_ids[mask]         (compact targets to match)
```

### Interface
- **New function**: `token_dropout(input_ids, target_ids, drop_rate=0.1) → (input_ids', target_ids')`
- **Integration point**: In training loop, after loading batch, before model forward
- **Env var**: `TOKEN_DROP_RATE=0.1`
- **Inference**: No dropout (drop_rate=0 when not training)

### Implementation Detail: Variable-Length Sequences
Token dropout produces different lengths per sequence in a batch. Options:
1. **Pad to max surviving length** — simple but wastes some compute
2. **Apply per-sequence with shared mask** — same mask across batch, all sequences have same length
3. **Apply before batching** — most efficient but requires data loader changes

**Recommended: Option 2** (shared mask) — simplest, no padding issues, same positions dropped across all sequences in batch. This also means RoPE positions are consistent across the batch.

### RoPE Interaction
With shared mask + compaction, the surviving tokens get new contiguous positions 0,1,2,...
This means RoPE positions SHIFT — token at original position 5 might get RoPE position 4.
This is intentional — it acts as a positional augmentation, forcing the model to rely on content not position.

### Parameter Budget
Zero — data augmentation only.

### Verifiable DoD
1. `len(output_ids) < len(input_ids)` when drop_rate > 0
2. `output_ids[0] == input_ids[0]` — first token always preserved
3. `len(output_ids) == len(output_targets)` — input and target lengths match
4. At drop_rate=0: output == input (no-op)
5. At inference (model.eval()): output == input (disabled)
6. **Timing test**: step_avg with dropout is ≤ step_avg without dropout (shorter sequences = faster)

---

## R2-7: Token Dropout + Rho-1 Selective Loss

### Mathematical Spec
```
Input side (token dropout):
  Same as R2-5

Output side (Rho-1 selective loss):
  logits = model(input_ids')
  confidence_i = max(logits[i]) - second_max(logits[i])     (logit margin per position)
  mask_loss = confidence_i < percentile(confidence, q)       (keep hardest q% of tokens)
  loss = CE(logits[mask_loss], targets[mask_loss]) / mask_loss.sum()
```

### Interface
- **New function**: `selective_loss(logits, targets, keep_ratio=0.8) → loss`
- **Combines with**: R2-5 token dropout
- **Env vars**: `TOKEN_DROP_RATE=0.1`, `SELECTIVE_LOSS_KEEP=0.8`

### Rho-1 Difficulty Proxy
- Original Rho-1 (NeurIPS 2024 best paper runner-up) uses a reference model to score difficulty
- Our simplified version uses the model's own logit margin (top1 - top2) as a zero-cost proxy
- High margin → model is confident → "easy" token → skip loss
- Low margin → model is uncertain → "hard" token → keep loss

### Parameter Budget
Zero.

### Verifiable DoD
1. `loss.shape == ()` — scalar
2. At keep_ratio=1.0: equivalent to standard CE loss (all tokens kept)
3. At keep_ratio=0.8: exactly 80% of positions contribute to loss
4. Gradient is non-zero only for kept positions
5. Combined with token dropout: effective training signal ≈ 0.9 × 0.8 = 72% of original tokens

---

## R2-8: Graduated Token Dropout (20%→0%)

### Mathematical Spec
```
drop_rate(step) = max_rate × (1 - step / total_steps)
                = 0.2 × (1 - step / total_steps)

At step 0:           drop_rate = 0.20 (20% dropped)
At step total/2:     drop_rate = 0.10 (10% dropped)
At step total:       drop_rate = 0.00 (0% dropped — clean data)
```

### Interface
- Same as R2-5, with step-dependent rate
- **Env vars**: `TOKEN_DROP_MAX_RATE=0.2`, `TOKEN_DROP_SCHEDULE=linear_decay`

### Rationale
Early training benefits from regularization (features still forming). Late training benefits from clean data (fine-tuning exact predictions). Linear decay provides smooth transition.

### Verifiable DoD
1. At step 0: drop rate ≈ 0.2
2. At final step: drop rate = 0.0
3. Rate is monotonically non-increasing over training

---

## R2-9: Cross-Layer Barlow Twins

### Mathematical Spec
```
For each adjacent layer pair (i, i+1):
  h_i = output of Block_i                              [B, T, D]
  h_{i+1} = output of Block_{i+1}                      [B, T, D]
  
  z_i = (h_i - μ_i) / σ_i                              (batch-standardize, per feature)
  z_{i+1} = (h_{i+1} - μ_{i+1}) / σ_{i+1}
  
  C = (z_i^T · z_{i+1}) / (B × T)                      (cross-correlation [D, D])
  L_BT = Σ_{i≠j} C²_{ij}                               (decorrelation loss)

Total aux loss: λ × Σ_pairs L_BT
```

### Interface
- **No new parameters** — loss-only modification
- **Training loop change**: Store each block's output, compute pairwise BT loss
- **Env vars**: `CROSS_LAYER_BT=1`, `CROSS_LAYER_BT_LAMBDA=0.005`

### Computational Cost
- `C` is `[512, 512]` = 262K elements per layer pair
- 10 adjacent pairs for 11 layers
- Matmul cost: `O(B×T×D² × num_pairs)` — could be significant
- **Optimization**: Subsample features (e.g., 128 random dims) → C is `[128,128]`

### Verifiable DoD
1. `cross_layer_bt_loss.shape == ()` — scalar
2. At init (random weights): loss should be small (layers are approximately uncorrelated)
3. Gradient propagates to all block parameters
4. With `CROSS_LAYER_BT=0`: no auxiliary loss (baseline behavior)
5. Training doesn't diverge with λ=0.005

---

## R2-11: Corrupted Context Training (10%)

### Mathematical Spec
```
During training only:
  1. Forward pass (no grad):    logits_pred = model(input_ids)
  2. Predictions:               pred_tokens = argmax(logits_pred, dim=-1)
  3. Corruption mask:           mask_i ~ Bernoulli(corruption_rate), mask[0] = 0
  4. Corrupted input:           input_ids'[i] = pred_tokens[i] if mask[i] else input_ids[i]
  5. Training forward pass:     logits = model(input_ids')
  6. Loss:                      CE(logits, original_targets)   ← targets unchanged
```

### Interface
- **New function**: `corrupted_context(model, input_ids, rate=0.1) → input_ids'`
- **Integration point**: Training loop, between data loading and training forward pass
- **Env var**: `CORRUPT_RATE=0.1`

### Computational Cost
- Extra forward pass (no backward) per training step
- Overhead: ~30-40% wall-clock time increase
- Fewer training steps in 10-min window: ~7000 vs ~9100

### Key Subtlety: Loss Targets
The loss targets remain the **original** ground truth tokens, NOT the corrupted ones. The model must learn to predict the correct next token even when some context tokens are wrong.

### Verifiable DoD
1. Corrupted input differs from original at ~10% of positions
2. Position 0 is never corrupted
3. At `CORRUPT_RATE=0`: output equals input (no-op)
4. At inference: output equals input (disabled)
5. Loss is computed against original targets (not corrupted)
6. Extra forward pass uses `torch.no_grad()` (no gradient accumulation)

---

## R2-12: Graduated Corruption (0%→20%→0%)

### Mathematical Spec
```
corruption_rate(step) = max_rate × sin(π × step / total_steps)
                      = 0.2 × sin(π × step / total_steps)

At step 0:           rate = 0.00 (clean — learn basics)
At step total/2:     rate = 0.20 (peak corruption — learn robustness)
At step total:       rate = 0.00 (clean — fine-tune)
```

### Interface
- Same as R2-11, with step-dependent rate
- **Env vars**: `CORRUPT_MAX_RATE=0.2`, `CORRUPT_SCHEDULE=sine`

### Rationale
- **Phase 1 (steps 0-30%)**: Clean data. Model learns basic next-token prediction.
- **Phase 2 (steps 30-70%)**: Corruption ramps up. Model learns robustness to imperfect context.
- **Phase 3 (steps 70-100%)**: Corruption decays. Model fine-tunes on clean data.

### Verifiable DoD
1. At step 0: corruption rate = 0.0
2. At step total/2: corruption rate ≈ 0.2
3. At final step: corruption rate ≈ 0.0
4. Rate follows sine curve (monotone increase then decrease)
