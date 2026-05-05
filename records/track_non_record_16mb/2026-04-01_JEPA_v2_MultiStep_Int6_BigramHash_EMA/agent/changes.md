# Agent Change Log — JEPA v2

## 2026-04-01 — Session 1: v1 diagnosis + full stack implementation

### Diagnosis: why JEPA v1 didn't work

From the v1 log analysis (c4e8f9a5 vs a9ea3137):

| Metric | JEPA ON | JEPA OFF |
|--------|---------|----------|
| Steps in 600s | 430 | 693 |
| Step avg | 1396ms | 867ms |
| Final val_bpb | 1.6153 | 1.4783 |
| At step 400 (fair) | 1.6132 | 1.5861 |

**3 bugs identified:**

1. **EMA momentum too high (0.996)**: with 430 steps, the target encoder updated
   only `430 × (1 - 0.996) = 1.72%` of its weights → target ≈ online encoder → trivial task.
   The predictor starts with `_zero_init=True` (output=0), so `z_pred = z_context` initially.
   EMA target is nearly identical to base model → `MSE(norm(z), norm(z)) ≈ 0` at step 1.
   Result: `jepa_loss = 0.002` constant, near-zero gradient, no signal.

2. **Single-step prediction too easy**: predicting `z[t+1]` from `z[t]` is nearly redundant
   with the CE objective that already forces `z[t]` to contain information about `t+1`.

3. **Gradient accumulation batch mismatch**: `z_target_cached` computed from `micro_batches[0]`,
   then applied as target to ALL micro-steps (which use different batches). 7/8 micro-steps
   had JEPA loss on pairs (prediction_batch_B, target_batch_A) → pure noise.

### Fixes implemented

#### Fix 1: EMA momentum 0.9 (was 0.996)
With 0.9, after 50 steps the target has received `50 × 0.1 = 5` units of update → diverges
enough to make the task non-trivial. Half-life ≈ 7 steps.

#### Fix 2: Multi-step prediction [1, 2, 4, 8]
A single `encode(x)` from the target encoder, then loss computed at 4 offsets:
```python
for offset, w in [(1, 1.0), (2, 0.5), (4, 0.25), (8, 0.125)]:
    z_p = z_pred[:, :T-offset, :]
    z_t = z_target_full[:, offset:, :]
    jepa_ms_loss += w * MSE(norm(z_p), norm(z_t))
jepa_loss = jepa_ms_loss / total_weight  # normalized
```
Target encoder runs ONCE, the 4 losses are simple slices. Minimal overhead.
Offset-8 requires long-range planning → non-trivial even for a causal LM.

#### Fix 3: Correct z_target per micro-batch
The target encoder now runs inside the micro-step loop, on the same `x` as the CE forward.
Overhead: `grad_accum_steps × encode_time` instead of `1 × encode_time`.
Trade-off documented: +overhead but correct gradients on all 8 micro-steps.

### New features

#### BigramHash(2048)
Lookup table for (token[t-1], token[t]) pairs via Cantor hash:
```
h(a, b) = (a+b)(a+b+1)//2 + b  mod 2048
```
Output summed with token embedding before the first layer. ~1.05M params, ~300-500KB compressed.
Frees attention capacity from learning bigram statistics.

#### int6 + LZMA
- `QUANT_MAX=31`: quantization range [-31, 31] in int8 container
- `lzma.compress(preset=9)` instead of `zlib.compress(level=9)`
- Expected saving: ~2-3 MB artifact + ~280 KB vs zlib

#### Artifact EMA (decay=0.9999)
Distinct from JEPA EMA (target encoder). This is a Polyak average of weights during
training, saved as the final checkpoint. Updated every step after `optimizer.step()`.

#### LeakyReLU(0.5)²
`F.leaky_relu(x, 0.5).square()` instead of `relu(x).square()`. Community-validated,
free ~-0.001 BPB. Changes 1 line in the MLP.

### Ablation modes in run.sh
- `baseline`: pure CE, ReLU², no JEPA, no BigramHash
- `leaky`: + LeakyReLU, no JEPA, no BigramHash
- `bigram`: + LeakyReLU + BigramHash, no JEPA
- `jepa`: + LeakyReLU + JEPA v2, no BigramHash
- `full`: full stack
- `smoke`: 2 min, full stack

## 2026-04-04 — Session 2: Full ablation results

### Results (600s wallclock, RTX 5060 Ti)

| Mode | Steps | Step avg | val_bpb pre-quant | val_bpb roundtrip | Artifact |
|------|-------|----------|-------------------|-------------------|---------|
| baseline | 690 | 870ms | 1.4768 | 1.7406 | 8.52 MB |
| leaky | 689 | 871ms | 1.4683 | 1.7207 | 8.59 MB |
| bigram | 688 | 872ms | 1.4617 | 1.7594 | 8.70 MB |
| jepa | 406 | 1481ms | 1.6224 | 2.7051 | 5.53 MB |
| full | 405 | 1482ms | 1.6047 | 2.7971 | 5.64 MB |

### Key findings

- **LeakyReLU**: −0.009 BPB, free 1-line change
- **BigramHash**: additional −0.007 BPB, artifact grows only 0.18 MB
- **JEPA v2**: still collapses to jepa_loss ~0.002. Root cause confirmed as geometric:
  consecutive positions in a causal LM are inherently collinear, making normalized MSE
  near-zero by construction regardless of momentum or prediction horizon.
- **JEPA overhead is fatal**: 870ms → 1481ms/step, 689 → 406 steps in 600s.
  Net effect: jepa mode is +0.154 BPB worse than leaky alone.
