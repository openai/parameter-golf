# Parameter Golf

> Trained on 1× NVIDIA H100 · 4800s per run · int8 + zlib submission format


### Change 1 — NorMuon (replaces vanilla Muon)

The `Muon` class was completely rewritten to implement NorMuon while keeping the same class name so nothing else in `main()` needed updating.

**What was added inside the optimizer step:**
- After Newton-Schulz orthogonalization, the per-row L2 norm of the update matrix is computed for each 2D weight
- An EMA (exponential moving average) of squared row norms is maintained in a new state buffer `row_norm_v` per parameter
- Each row of the update is divided by `sqrt(its EMA + eps)` — this balances how much each neuron gets updated, preventing dominant neurons from taking over training
- Decoupled weight decay (`p.mul_(1 - lr * weight_decay)`) is applied before each gradient step to prevent weight norms from growing over long runs

**New hyperparameters:**
- `muon_weight_decay = 0.1` — controls L2 regularisation strength on matrix params
- `beta2 = 0.95` — controls EMA decay of row norms

**Why:** Vanilla Muon equalises singular values but leaves per-neuron update magnitudes highly non-uniform. NorMuon fixes this, reported to reach target loss ~21% faster.

---

### Change 2 — q_gain initialisation: 1.5 → 1.0

`QK_GAIN_INIT` default was lowered from `1.5` to `1.0` in `Hyperparameters`. This flows into `CausalSelfAttention` where `q_gain` is initialised.

**Why:** After QK-norm, query vectors already have unit RMS. Starting `q_gain` at 1.5 immediately re-inflates the attention logit scale at step 0, which can cause early loss spikes. Starting at 1.0 is a neutral baseline — the model learns to scale up as needed. QK-norm keeps things stable as `q_gain` grows.

---

### Change 3 — fake_quant_int6 (QAT activation quantization)

A new function `fake_quant_int6()` was added and applied in two places:
- Inside `CausalSelfAttention.forward()` — on the attention output `y` before the projection
- Inside `MLP.forward()` — on the relu output before squaring and projection

**How it works (straight-through estimator):**
- Forward pass: quantizes activations to 6-bit integers in range `[-31, 31]` using a per-tensor abs-max scale
- Backward pass: gradient flows through unchanged (the `.detach()` trick makes quantization invisible to the optimizer)

**Why:** The original baseline trained on clean fp32 activations but exported as int8, creating a train/eval distribution gap. With `fake_quant_int6`, the model learns weights that are inherently robust to quantization noise, making the int8+zlib roundtrip BPB closer to training-time BPB.

**Escape hatch:** Set `FAKEQUANT_EVAL_SKIP=1` to skip fake_quant during eval only (reverts to original behaviour for ablation testing).

---

### Change 4 — num_layers: 9 → 12

`NUM_LAYERS` default increased from `9` to `12`.

**Impact on U-Net structure:**

| Component | Before | After |
|---|---|---|
| Encoder layers | 4 | 6 |
| Decoder layers | 5 | 6 |
| Skip connections | 4 | 6 |

More layers means more representational depth and more skip connections carrying encoder features into the decoder. At `model_dim=512` and `mlp_mult=2`, 12 layers still fits within the 16MB submission budget.

---

## Results

| Metric | Seed 42 | Seed 1337 |
|---|---|---|
| Val loss | 2.0781 | 2.0772 |
| Val BPB | 1.2308 | 1.2302 |
| Val loss (exact) | 2.07812564 | 2.07715473 |
| Val BPB (exact) | 1.23078306 | 1.23020803 |
| Eval time | 15,687 ms | 15,741 ms |
| Peak VRAM allocated | 13,516 MiB | 13,516 MiB |
| Peak VRAM reserved | 13,742 MiB | 13,798 MiB |
| Serialized model (raw) | 89,284,811 B | 89,284,811 B |
| Serialized model (int8+zlib) | 15,337,366 B | 15,334,608 B |
| Total submission size (int8+zlib) | 15,391,075 B | 15,388,319 B |
| Compression ratio | 3.93× | 3.93× |

Both seeds trained identically on 1× H100 for 4800s. Seed 1337 is marginally better on all accuracy metrics.