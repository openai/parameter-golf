# FX_Wing_Delta — Flow Instructions + DeltaNet

## The Core Problem We're Solving

FX_Wing introduced content-derived loop instructions to fix Frugendorff's gradient
conflict. It worked architecturally but had a critical flaw: the instructions were
a **perturbation** — computed once from the encoder output before any loop runs.

By loop 3, the activations `x` have drifted far from the encoder state that generated
the instructions. The correction is now pointing at a target that no longer exists.

The quantization result confirmed this: +2.93 BPB gap at 450 steps. The shared weights
serving 4 different activation distributions, with instructions that don't adapt to
those distributions, cannot be quantized cleanly.

---

## Hypothesis H0 — Flow > Perturbation

**Claim:** Recomputing the loop instruction from the CURRENT `x` at each loop iteration,
rather than pre-planning all instructions from `x_enc`, will:

1. **Reduce gradient conflict** — `∂L/∂W_inst = Σ_k δ_k ⊗ x_{k-1}ᵀ` now has different
   `x_{k-1}` per loop (not the same `x_enc` for all). Outer products less likely to cancel.

2. **Reduce quantization gap** — At inference, the instruction for loop k is computed
   from quantization-distorted activations. The instruction implicitly compensates for
   the quantization error, partially restoring the signal. Self-healing quant.

3. **Simpler architecture** — `loop_inst_proj` projects `model_dim → inst_dim` (not
   `model_dim → K*inst_dim`). Less parameters in the instruction path, computed K times
   rather than 1 time with K outputs. Equivalent parameter count, better gradient flow.

**Architecture change (one line):**
```python
# FX_Wing (perturbation — planned before loops):
inst = proj(x_enc)                     # computed once
x_loop = x + up[k](inst[:,:,k,:])     # static correction

# FX_Wing_Delta (flow — responsive at each loop):
inst_k = up[k](proj(x))               # recomputed from CURRENT x
x_loop = x + inst_k                   # adaptive correction
```

**Expected result:** Flow instructions achieve lower int6 roundtrip BPB than FX_Wing
at the same training budget. Specifically, quant gap < 0.5 BPB (vs +2.93 in FX_Wing).

---

## Hypothesis H1 — DeltaNet adds iterative refinement

**Claim:** With flow instructions reducing gradient conflict, the crawler loops now
produce genuinely different activations per pass. DeltaNet's associative memory state
`S` accumulates the pattern associations across these genuinely-different passes,
providing cumulative refinement that a stateless loop cannot.

DeltaNet update rule: `S_t += β_t * outer(v_t - S_t @ k_t, k_t)`

Each pass reads from S (what previous loops learned about this token context) and
writes corrections back. With static instructions (FX_Wing), loop outputs are similar
enough that S accumulates nothing useful. With flow instructions, each loop produces
a distinct representation, giving S meaningful content to accumulate.

**Expected result:** FX_Wing_Delta (flow + DeltaNet) > FX_Wing_Delta (flow only) >
FX_Wing (static) on val_bpb at equivalent training compute.

---

## Hypothesis H2 — File size advantage is real

**Claim:** FX_Wing_Delta achieves comparable BPB to flat SOTA (1.1129) with
significantly smaller artifact.

Weight sharing math:
- Crawler block stored ONCE, run LOOPS=4 times
- 4 flat + 1 crawler×4 = 8 effective blocks, ~9.5M unique params stored
- Flat equivalent: 8 blocks = ~14M unique params stored
- Structural compression: ~35% smaller artifact for same effective depth

**Metric to watch:** `int6_bpb / artifact_MB` — the quality-per-byte ratio.
- SOTA Green: ~1.113 / 8.6 MB = 0.129
- FX_Wing_Delta target: ~1.15 / 4.5 MB = **0.256** (2× better ratio)

If achieved, this is a genuinely novel result: better compression efficiency than
any flat architecture in the competition.

---

## Ablation Ladder

### B0 — FX_Wing baseline (already run)
```
USE_CRAWLER=1  INST_DIM=32  CRAWLER_LOOPS=4  DELTA_NET_HEADS=0
```
Static instructions, no DeltaNet. Reference point. Result: +2.93 BPB quant gap at 450 steps.

### B1 — FX_Wing_Delta: flow only (DELTA_NET_HEADS=0)
```
USE_CRAWLER=1  INST_DIM=32  CRAWLER_LOOPS=4  DELTA_NET_HEADS=0
```
Flow instructions, no DeltaNet. Isolates the instruction architecture change.
**Key test**: does quant gap shrink vs B0?

### B2 — FX_Wing_Delta: flow + DeltaNet (main hypothesis)
```
USE_CRAWLER=1  INST_DIM=32  CRAWLER_LOOPS=4  DELTA_NET_HEADS=2
```
Full architecture. This is what run.sh runs.
**Key test**: does DeltaNet improve val_bpb vs B1?

### B3 — Flat control (A1 from original FX_Wing plan)
```
USE_CRAWLER=0
```
Same training config, flat blocks only, no crawler. Establishes whether the crawler
architecture is buying anything at all vs equivalent flat capacity.

### B4 — CRAWLER_LOOPS=2 (quant stress reduction)
```
CRAWLER_LOOPS=2  DELTA_NET_HEADS=2
```
If quant gap is still problematic at LOOPS=4, reduce to 2. Less compression gain
but more quantization-friendly. Decision gate: if int6 gap > 0.3 BPB, run B4.

---

## Follow-up: Per-Loop Quantization Scales

If flow instructions reduce but don't eliminate the quant gap, the next step is
per-loop GPTQ scales: same int8 quantized weights, K separate dequantization scales
(one per loop), each calibrated against that loop's specific activation distribution.

```python
# At inference, loop k uses scale_k instead of a single shared scale
W_approx_k = W_int8 * scale_k
```

This is a zero-retraining fix (export path only). Combined with flow instructions,
it should bring the quant gap to near-flat-model levels.

---

## Decision Criteria

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| int6 gap < 0.2 BPB AND val_bpb ≤ 1.15 | Full win. Push to 8×H100. | Submit |
| int6 gap < 0.5 BPB, val_bpb competitive | Gap improved, not fixed. | Add per-loop scales |
| int6 gap > 1.0 BPB still | Flow not sufficient. | LOOPS=2 + per-loop scales |
| val_bpb worse than flat control (B3) | Crawler adds noise. | Park FX_Wing_Delta |
