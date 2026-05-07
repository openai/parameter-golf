# exp107: SAM Inner Loop for Meta-TTT

**Parent**: exp106 — 11L XSA-all · BigramHash 4096×64 pos-conditional · VE7-10
            · FOMAML every=4 cross-chunk + delta-loss + MetaSGD · SGD+cosine TTT
            · int6 GPTQ+lzma (float-path legal_ttt **1.11469**)

**Change**: Replace MetaSGD (C) with Sharpness-Aware Minimization (D) in the
FOMAML inner loop. No architecture change.

**Verdict**: ❌ **SAM hurts — discard.** TTT delta invariant at ~0.023 bpb.
Absolute legal_ttt 1.1190 is worse than exp106's float-path 1.11469. The TTT
ceiling is architecture-limited, not inner-loop-optimizer-limited.

---

## Results

| Metric | exp105a (no meta-TTT) | exp106 (MetaSGD) | **exp107 (SAM)** |
|--------|----------------------|------------------|------------------|
| Training steps | ~6892 | 6686 | **6597** |
| Float val_bpb | 1.1353 | 1.1377 | **1.1384** |
| Int6 roundtrip val_bpb | 1.1396 | 1.1416 | **1.1424** |
| Legal TTT val_bpb | 1.11624 | 1.11469† | **1.11898** |
| TTT delta (int6) | −0.0234 | ~−0.023 | **−0.0234** |
| Peak GPU memory | ~23 GB | 31.7 GB | **32.4 GB** |
| Per-step time | ~727 ms | ~718 ms | **~728 ms** |
| Submission size | — | — | **15.88 MB** |

†exp106 legal_ttt measured on float-path (int6 canonical path partial at 80%: 1.11800)

**TTT delta is identical (−0.023 bpb) across all three formulations.**
SAM changed the gradient direction in the inner loop — but the result at eval time
is indistinguishable from vanilla SGD inner loop.

---

## 1. Motivation (Pre-run)

### Why replace MetaSGD with SAM?

exp106's three-way analysis revealed two things:

1. **MetaSGD failed**: All 66 learned per-layer LR scales converged to their 1.0
   initialization. The meta-gradient signal (1 step per 4, at ~30% of main gradient
   magnitude) was too weak to drive per-layer differentiation. MetaSGD cost +8.6 GB
   peak memory and −334 training steps for zero benefit.

2. **Bank curvature is invariant**: Condition numbers (1.03–1.38), effective ranks
   (22/11), and energy distributions are identical across all three models (exp101,
   exp105a, exp106). The TTT delta is ~0.023 bpb regardless of meta-TTT formulation.

But all three experiments used **vanilla SGD** in the inner loop. The gradient
DIRECTION was always the same — only the meta-objective and step size varied.

**SAM changes the gradient direction itself.** Instead of descending along `∇L(θ)`,
SAM descends along `∇L(θ + ε)` where `ε = ρ · ∇L / ‖∇L‖` is a small ascent
step. This gradient points toward **flatter minima** — regions where small
perturbations don't increase loss. If the ~0.023 bpb TTT ceiling is determined by
local curvature, SAM's explicit flatness-seeking might change it.

### Why SAM might work where MetaSGD didn't

| Property | MetaSGD | SAM |
|---|---|---|
| What it changes | Step SIZE per layer (scalar scale) | Gradient DIRECTION (via ascent perturbation) |
| Free parameters | 66 (need to be learned) | 0 (rho is a fixed hyperparameter) |
| Signal requirement | Needs meta-gradient to push 66 params away from init | Operates on each gradient independently |
| Memory cost | +8.6 GB (gradient graph for differentiable non-leaf) | +2 GB (one extra forward pass of activations) |
| Theoretical target | Per-layer adaptation speed differentiation | Flatness of the adapted banks |

---

## 2. Maths

### Standard inner loop (exp101/exp105a/exp106):

$$
\theta' = \theta - \alpha \cdot \nabla_\theta \mathcal{L}(\theta;\, \mathcal{B}_A)
$$

### SAM inner loop (exp107):

Step 1 — Ascent perturbation:

$$
\hat{\epsilon} = \rho \cdot \frac{\nabla_\theta \mathcal{L}(\theta;\, \mathcal{B}_A)}
                                  {\|\nabla_\theta \mathcal{L}(\theta;\, \mathcal{B}_A)\|}
$$

Step 2 — Sharpness-aware gradient (gradient at the perturbed point):

$$
g_\text{SAM} = \nabla_\theta \mathcal{L}(\theta + \hat{\epsilon};\, \mathcal{B}_A)
$$

Step 3 — Descent using SAM gradient:

$$
\theta' = \theta - \alpha \cdot g_\text{SAM}
$$

The outer loop is unchanged from exp106:

$$
\mathcal{L}_\text{meta} = (w_\text{post} + w_\Delta) \cdot \mathcal{L}(\theta';\, \mathcal{B}_B)
                         - w_\Delta \cdot \mathcal{L}(\theta;\, \mathcal{B}_B)
$$

---

## 3. Implementation

### Changes to `meta_ttt_step` (the only modified function):

```python
# After computing vanilla gradient g and applying freeze mask...
if sam_on:
    # Joint gradient norm across all 4 banks
    grad_norm = (g_qo.float().norm()**2 + g_kv.float().norm()**2 +
                 g_up.float().norm()**2 + g_down.float().norm()**2
                 ).sqrt().clamp(min=1e-12)

    # Ascent perturbation
    with torch.no_grad():
        scale = rho / grad_norm
        qo_pert = (qo_in.detach() + scale * g_qo).requires_grad_(True)
        kv_pert = (kv_in.detach() + scale * g_kv).requires_grad_(True)
        up_pert = (up_in.detach() + scale * g_up).requires_grad_(True)
        down_pert = (down_in.detach() + scale * g_down).requires_grad_(True)

    # SAM gradient at the perturbed point
    loss_pert = base_model.forward_with_banks(x_inner, y_inner, *_pert)
    g_sam = torch.autograd.grad(loss_pert, [*_pert])

    # Use g_sam instead of g for the adapted banks
    upd = bank.detach() - lr * g_sam
```

### Removed from exp106:
- `meta_sgd_{qo,kv,up,down}` nn.Parameters from `GPT.__init__`
- MetaSGD optimizer param group
- MetaSGD export filter and strict-load hotfix

### New env vars:
- `META_TTT_SAM_ENABLED=1` — enable SAM inner loop
- `META_TTT_SAM_RHO=0.05` — perturbation radius
- `META_TTT_SAM_ADAPTIVE=0` — 0=vanilla SAM, 1=adaptive (scale ε by |param|)

---

## 4. Budget Analysis (Predicted vs Actual)

### Memory

| Component | exp106 | exp107 predicted | **exp107 actual** |
|---|---|---|---|
| MetaSGD gradient graph | +8.6 GB | 0 | **0** |
| SAM extra forward activations | 0 | ~2.0 GB | **~2.7 GB** |
| SAM perturbation tensors | 0 | ~0.1 GB | **~0.1 GB** |
| **Peak** | **31,695 MiB** | **~25,200 MiB** | **32,397 MiB** |

**The memory prediction was wrong.** SAM's extra forward pass holds activations for
ALL 11 layers simultaneously (needed for the backward through the perturbed forward),
which costs more than the 2 GB estimated. The MetaSGD gradient graph was surprisingly
efficient at storing only the parameter-level graph nodes, not full activations.
Net result: exp107 peak memory is +702 MiB HIGHER than exp106, not −6.5 GB lower.

### Compute

| Metric | exp106 | exp107 predicted | **exp107 actual** |
|---|---|---|---|
| Per-step time | ~718 ms | ~706 ms | **~728 ms** |
| Steps in 4800s | 6686 | ~6800 | **6597** |

SAM's extra activation memory caused more GPU memory pressure, slightly reducing
throughput. exp107 ran **89 fewer steps** than exp106 — the opposite of the prediction.

---

## 5. Decision Thresholds

Compare TTT delta against exp105a's baseline of −0.02331 bpb:

| TTT delta | Verdict |
|---|---|
| < −0.026 (>10% better) | SAM genuinely helps — integrate into future runs |
| −0.026 to −0.024 | Marginal — try rho sweep {0.01, 0.02, 0.1, 0.2} |
| −0.024 to −0.023 | Same ceiling — architecture-limited hypothesis confirmed |
| **> −0.023 (actual: −0.0234)** | **SAM hurts — discard** |

**Verdict**: Same ceiling confirmed. TTT delta = −0.023 bpb across all 4 experiments
(exp101, exp105a, exp106, exp107). The ceiling is set by the bank architecture
(rank × dim), not by inner-loop optimizer choice.

---

## 6. Post-Hoc Weight-Space Analysis

Four-way principal-angle + midpoint analysis (exp101, exp105a, exp106, exp107):

| Pair | Bank cosine | Midpoint ratio | Interpretation |
|---|---|---|---|
| exp105a ↔ exp101 | ~0.05 | ~0.91 | Different basins (different seeds) |
| exp106 ↔ exp101 | ~0.05 | ~0.91 | Different basins |
| **exp107 ↔ exp106** | **0.2025** | **0.839** | **Same basin — mildest perturbation** |
| exp107 ↔ exp105a | ~0.05 | ~0.91 | Different basins |

exp107 is the most similar to exp106 of any pair in the series. SAM barely shifted the
trained weights — it is the smallest perturbation in all four experiments. This is
consistent with SAM's rho=0.05 being a tiny fraction of the bank norms (~2.7).

---

## Run

```bash
bash records/phase3/exp107_sam-inner-metattt_from_exp106/run.sh
```

Hardware: **1×H100 80 GB SXM**, `MAX_WALLCLOCK_SECONDS=4800` (80-minute cap).
Iso-compute with the competition's 8×H100 @ 10-min budget.

**Actual completion**: 6597/7500 steps (wallclock cap), seed=42.
