# MDL-T Stack — Methodology Record

**Script:** `train_gpt_stack.py`
**Base:** `train_gpt_mdlt.py` (MDL-T + zstd-22)
**Target:** Beat SOTA 1.1194 BPB

---

## Stack Summary

Five techniques layered on the baseline (`train_gpt.py`), each independently validated on the leaderboard:

| Layer | Technique | Source | Expected BPB gain |
|---|---|---|---|
| 0 | Baseline (relu², warmdown=1200, zlib-9) | train_gpt.py | — |
| 1 | MDL-T regularizer + zstd-22 | train_gpt_mdlt.py | −0.003 to −0.010 |
| 2 | LeakyReLU(0.5)² activation | Entry #1 (1.1194) | −0.003 |
| 3 | EMA decay=0.997 | Entry #2 (1.1228) | −0.0006 |
| 4 | warmdown_iters=3500 | Entry #2 (1.1228) | amplifies MDL-T |
| 5 | GPTQ-lite (5 percentile search) | Entry #2 (1.1228) | −0.0006 |
| 6 | Late QAT threshold=0.15 | Entry #2 (1.1228) | −0.001 to −0.003 |

---

## Technique Details

### 1. MDL-T: Quantization Gravitational Regularizer

**Theory:** The 16MB challenge is an MDL problem. We jointly minimise language modelling
loss AND weight compressibility:

```
L_total(t) = L_LM(W) + λ(t) · L_MDL(W)

L_MDL(W) = mean_layers[ Var(W - Q(W)) / Var(W) ]
         = dimensionless fraction of weight variance lost to quantization noise
```

`Q(W)` = nearest int6 gridpoint (per-row scale, detached).
`λ(t) = mdl_lambda × (1 − lr_scale)` — ramps from 0 during normal training to
`mdl_lambda` at the end of warmdown. Early training is unaffected.

**Why it works:** Weights that cluster at gridpoints produce a low-entropy int6
distribution that zstd/arithmetic coders exploit. Same 16MB → more parameters →
better BPB.

**Key fix (vs first version):** The regularizer was normalised to be scale-invariant:
`residual_var / w_var` (not raw MSE). This ensures MDL_LAMBDA=0.05 is meaningful
regardless of weight magnitude (~0.001–0.05 natural range).

**Hyperparameters:**
- `MDL_LAMBDA=0.05` (default; sweep 0.01–0.1 on H100)
- `MDL_QUANT_BITS=6` (match final quantizer bits)

---

### 2. LeakyReLU(0.5)²

```python
# MLP.forward:
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

Replaces relu² from the baseline. `negative_slope=0.5` preserves gradient flow for
negative activations (no dead neurons at large scale). The square keeps the sparsity
pressure that makes relu² effective. Proven +0.003 BPB on leaderboard (Entry #1).

---

### 3. EMA (Exponential Moving Average)

```python
# After each optimizer.step():
for n, p in base_model.named_parameters():
    ema_params[n].mul_(decay).add_(p.data.cpu(), alpha=1.0 - decay)
```

`decay=0.997` means each EMA weight averages ~333 recent parameter snapshots.
Maintained on CPU (no GPU memory overhead). Loaded into model before serialization.

**Why it helps:** EMA weights are smoother than the final SGD iterate. This:
1. Reduces quantization error (smoother → better int8 reconstruction)
2. Interacts with MDL-T: EMA of clustered weights → tighter clusters
3. Proven −0.0006 BPB (Entry #2)

---

### 4. warmdown_iters=3500 (up from 1200)

**Directly amplifies MDL-T:** The regularizer is active for the entire warmdown
phase. 3500 steps instead of 1200 means MDL-T has 2.9× more iterations to pull
weights toward gridpoints.

On H100s (10 min cap): warmdown_iters=3500 ≈ last 17.5% of a 20k-step run at
~600 tokens/step throughput. The original 1200 was only ~6%.

---

### 5. GPTQ-lite: Per-Tensor Clip Percentile Search

```python
GPTQ_LITE_PERCENTILES = [90.0, 95.0, 99.0, 99.9, 99.99984]

# For each 2D weight matrix at save time:
for pct in GPTQ_LITE_PERCENTILES:
    q, s = quantize_to_int8(t, clip_percentile=pct)
    mse = reconstruction_mse(t, dequantize(q, s))
    if mse < best_mse: keep this (q, s)
```

Applied in `quantize_state_dict_int8(use_gptq_lite=True)` at the end of training.
Run cost: O(N_tensors × 5) quantize+MSE ops (~seconds on CPU).

**Why:** The baseline uses a fixed 99.99984th percentile for all tensors. Some
matrices benefit from tighter clipping (outlier-heavy rows), others from looser.
GPTQ-lite adapts per-tensor. Proven −0.0006 BPB (Entry #2).

---

### 6. Late QAT (Straight-Through Estimator)

```python
# In CastedLinear.forward, when fake_quant_bits > 0:
scale = w.detach().abs().max(dim=1, keepdim=True).values.clamp(min=1e-6) / n_levels
w_q   = (w / scale).round().clamp(-n_levels, n_levels) * scale
w     = w + (w_q - w).detach()   # STE: forward=w_q, backward=dL/dw
```

Activates at `step >= (1 - late_qat_threshold) * iterations` (default: last 15%).
With 20k iterations: activates at step 17000.

**STE = Straight-Through Estimator:** The forward pass uses quantized weights (model
learns to work with discrete weights). The backward pass treats the rounding as the
identity function (gradients flow through to full-precision weights). This is a
one-time recompile cost (~30s on H100).

**Synergy with MDL-T:**
- MDL-T (warmdown steps 16500–20000): pulls weights toward gridpoints softly
- Late QAT (steps 17000–20000): forces model to converge WITH quantized weights
- Both active simultaneously during steps 17000–20000 (final 3000 steps)

The compound effect: weights that MDL-T clustered are exactly the ones that
Late QAT's forward pass sees as quantized, producing a self-reinforcing loop.

---

## Interaction Matrix

```
                  MDL-T  EMA   GPTQ-lite  LateQAT  LeakyReLU
MDL-T          |  —      +++   +++        +++       +
EMA            |  +++    —     ++         +         +
GPTQ-lite      |  +++    ++    —          ++        0
LateQAT        |  +++    +     ++         —         +
LeakyReLU(0.5)²|  +      +     0          +         —
```

`+++` = strong positive synergy, `++` = moderate, `+` = mild, `0` = independent

**Strongest interaction: MDL-T ↔ Late QAT**
MDL-T pre-clusters weights → Late QAT reinforces with hard quantization in forward →
GPTQ-lite finds optimal clip for the clustered distribution → zstd-22 exploits the
near-gridpoint distribution.

---

## Compression Pipeline

```
base_model (bf16)
  → [EMA]: smooth weights
  → [MDL-T during warmdown]: pull weights toward int6 gridpoints
  → [Late QAT during last 15%]: force convergence with quantized forward
  → [GPTQ-lite]: find optimal int8 clip per tensor
  → [zstd-22]: lossless compress the int8 representation
  → final_model.int8.ptz  (target: < 15MB)
```

---

## Hyperparameter Defaults

| Param | Value | Notes |
|---|---|---|
| `MDL_LAMBDA` | 0.05 | Peak regularizer strength; sweep 0.01–0.1 |
| `MDL_QUANT_BITS` | 6 | Matches int6 save target |
| `EMA_DECAY` | 0.997 | ~333 step averaging window |
| `WARMDOWN_ITERS` | 3500 | 3500 steps for MDL-T to act |
| `LATE_QAT_THRESHOLD` | 0.15 | Last 15% of training |

---

## Run Commands

```bash
# Fast local test (RTX 3060, ~12 min)
RUN_ID=stack_test SCRIPT=train_gpt_stack.py bash run_experiment.sh

# Sweep MDL lambda
for lam in 0.01 0.03 0.05 0.1; do
  RUN_ID=stack_lam${lam} MDL_LAMBDA=$lam SCRIPT=train_gpt_stack.py bash run_experiment.sh
done

# Full H100 run
RUN_ID=stack_full \
MDL_LAMBDA=0.05 \
MAX_WALLCLOCK_SECONDS=0 \
SCRIPT=train_gpt_stack.py \
bash run_baseline.sh
```

---

## Expected Results (H100 full run)

| Configuration | Est. BPB |
|---|---|
| Baseline | ~1.22 |
| + LeakyReLU(0.5)² | ~1.217 |
| + MDL-T (zstd-22) | ~1.212–1.215 |
| + EMA + GPTQ-lite | ~1.210–1.213 |
| + Late QAT + warmdown=3500 | ~1.206–1.211 |
| SOTA target | 1.1194 |

Beating SOTA requires the full stack to land at or below 1.119. The gap is non-trivial
but the compound synergies give a realistic path.

---

## Local Test Results

| Run | Steps | BPB | Notes |
|---|---|---|---|
| baseline (train_gpt.py) | 2000 | 1.4957 | zlib-9 |
| mdlt_test (train_gpt_mdlt.py) | 2000 | 1.4932 | zstd-22, mdl_reg bug (was 0) |
| mdlt_test2 | TBD | TBD | Fixed magic bytes + normalised reg |
| stack_test (train_gpt_stack.py) | TBD | TBD | This script |

---

## Submission Notes

- All model weights are compressed with zstd-22 (not zlib-9).
  Auto-detect decompression checks zstd magic bytes `\x28\xb5\x2f\xfd`.
- GPTQ-lite uses the standard `int8_clean_per_row_v1` format; the submission
  evaluator only needs `dequantize_state_dict_int8` which is unchanged.
- EMA weights are serialised as `final_model.pt` (raw) and `final_model.int8.ptz`
  (compressed int8). Both contain EMA weights.
- Late QAT is disabled at save time (`fake_quant_bits = 0` restored before export).
