# Quantization-Gate Correlation Analysis (2026-03-23)

## Summary

Checkpoint analysis of run3 (1.1496 BPB) reveals that Gated Attention (GA) learns to
compensate for quantization damage — heads and layers with higher int6 quantization error
have lower gate openings. But GA's own weights are among the most fragile to quantization,
creating a paradox: the correction mechanism is itself being corrupted.

## Method

Analyzed `final_model_no_ttt_20260323_142955.pt` (run3, 27.48M params, 11 layers).
Simulated int6 per-row quantization on all 2D weight matrices and measured:
- Per-head MSE for Q/K/V/proj projections
- Per-layer total quantization damage
- Correlation with learned GA gate openings (sigmoid of bias)

## Key Findings

### 1. GA compensates for quantization damage (corr = -0.43)

Layers with higher quantization damage have lower mean gate openings:

| Layer | Quant Damage | Gate Mean | Direction |
|-------|-------------|-----------|-----------|
| 0 | 0.0000563 (highest) | 0.4993 (low) | Dampened |
| 3 | 0.0000321 (low) | 0.6068 (highest) | Open |
| 10 | 0.0000395 (high) | 0.4929 (lowest) | Dampened |

**Per-head correlations are even stronger in deep layers:**
- Layer 8 c_q: **-0.70** (high-damage heads get closed)
- Layer 10 proj: **-0.75**
- Layer 0 c_q: **-0.61**
- Layer 9 c_q: **-0.64**

The pattern is strongest in Q projections and the deepest layers where the model
is most sensitive to weight perturbation.

### 2. GA's own weights are the most fragile (paradox)

The `attn_gate.weight` tensors (8×512) rank among the TOP damaged tensors:

| Rank | Tensor | Rel MSE | Shape |
|------|--------|---------|-------|
| 2 | blocks.10.attn.attn_gate.weight | 0.001896 | 8×512 |
| 4 | blocks.9.attn.attn_gate.weight | 0.001661 | 8×512 |
| 6 | blocks.3.attn.attn_gate.weight | 0.001482 | 8×512 |
| 7 | blocks.2.attn.attn_gate.weight | 0.001345 | 8×512 |
| 9 | blocks.6.attn.attn_gate.weight | 0.001294 | 8×512 |

These 8×512 matrices have very high relative quantization error because they're small
(only 4096 params each) and their values are concentrated in a narrow range. Int6 per-row
quantization with only 8 rows can't capture their distribution well.

**The paradox:** GA learns quant-damage correction, but quantization destroys the correction
itself. If we keep `attn_gate` weights in fp32 (exempt from int6), the correction survives.
Cost: 11 layers × 8 × 512 × 4 bytes = 176KB. Negligible.

### 3. c_k weights are the most damaged attention tensors

Key projections dominate the damage ranking:

| Rank | Tensor | Rel MSE |
|------|--------|---------|
| 1 | blocks.10.attn.c_k.weight | 0.002264 |
| 3 | blocks.9.attn.c_k.weight | 0.001872 |
| 5 | blocks.7.attn.c_k.weight | 0.001597 |
| 11 | blocks.0.attn.c_k.weight | 0.001278 |
| 16 | blocks.3.attn.c_k.weight | 0.001236 |
| 17 | blocks.1.attn.c_k.weight | 0.001228 |

c_k consistently has ~1.5-2x the relative MSE of c_q and c_v at the same layer. This is
likely because key vectors interact multiplicatively with queries in attention — small
perturbations in K get amplified through the softmax. Our per-layer LR gives mlp.proj 1.5x
and mlp.fc 0.7x, but doesn't differentiate between attention projections. **c_k should get
elevated treatment** — either higher LR during training, or int7/fp16 during quantization.

### 4. Value Residual is barely used

| Layer | v0 weight | v_current weight |
|-------|-----------|-----------------|
| 0 | 0.0620 | 0.0620 |
| 1 | 0.1058 | 0.5071 |
| 3 | 0.0137 | 0.8873 |
| 6 | 0.0140 | 1.1723 |
| 10 | 0.0666 | 0.2456 |

v0 weight is near-zero (0.01-0.10) for all layers. The model learned to mostly ignore the
layer-0 V cache. Value Residual provides minimal benefit in our architecture. The 22 params
are negligible but the feature adds code complexity.

### 5. resid_mix reveals "what's new" encoding

| Layer | x weight | x0 weight | Interpretation |
|-------|----------|-----------|----------------|
| 0 | 0.30 | +0.30 | Equal blend |
| 2 | 0.40 | -0.09 | Subtract embedding |
| 6 | 0.49 | -0.02 | Mostly current |
| 10 | 0.34 | +0.01 | Mostly current |

Middle layers learn **negative x0 weights** — they subtract the initial embedding. This is
naturally learning a form of differential/novelty encoding (similar to DG Attention from
PR #542). The model discovers that "what changed since embedding" is more useful than raw
content for middle layers.

### 6. Control tensor scaling patterns

`attn_scale` and `mlp_scale` show clear depth-dependent patterns:

- `attn_scale`: increases from 0.25 (L0) to 0.39 (L7), then drops to 0.33 (L10)
- `mlp_scale`: increases from 0.21 (L0) to 0.43 (L6), then sharply drops to 0.15 (L10)
- Combined with `ln_scale_factor = 1/sqrt(layer+1)`, the effective contribution of
  each layer is carefully calibrated. Layer 10's MLP is barely contributing (0.15 scale).

## Proposed Architectural Changes

### A. Keep attn_gate in fp32 during quantization (HIGH PRIORITY)

Add `attn_gate.weight` and `attn_gate.bias` to the CONTROL_TENSOR_NAME_PATTERNS list
so they're kept in fp32 instead of int6. This preserves GA's quant-damage correction.

- Cost: 176KB artifact increase (11 × (8×512 + 8) × 4 bytes)
- Expected: -0.002 to -0.005 BPB (the correction factors survive quantization)
- Risk: Very low — purely quantization-time change

### B. Per-key elevated LR or int7 for c_k (MEDIUM PRIORITY)

c_k weights have ~2x the quantization damage of c_q/c_v. Two options:

**Option B1: Training-time** — add `CK_LR_MULT=1.5` to give c_k weights higher LR,
similar to how mlp.proj gets 1.5x. This trains the model to develop more
quantization-robust key representations.

**Option B2: Quantization-time** — force c_k to int7 (63 levels instead of 31) via
GradQuant override. Cost: ~20% more bytes for c_k tensors across 11 layers.
c_k is 256×512 = 131K params/layer × 11 = 1.44M params total. At int7 vs int6,
~180KB extra.

These stack: B1 makes c_k more robust, B2 preserves more precision for whatever remains.

### C. Investigate removing Value Residual (LOW PRIORITY)

VR contributes near-zero v0 blending. Removing it saves 22 params (negligible) but
simplifies the forward path and removes a potential source of quantization interaction.
Test by setting `VALUE_RESIDUAL=0` and comparing.

## Verification

Run the combined config on 8xH100:
1. Baseline: run3 config (1.1496)
2. +A: fp32 attn_gate (expect improvement from preserved correction)
3. +A+B1: fp32 attn_gate + c_k LR 1.5x (expect further improvement)
4. Compare pre-quant and post-quant gaps to verify the mechanism

The analysis script is at `experiments/analyze_quant_gates.py`.
