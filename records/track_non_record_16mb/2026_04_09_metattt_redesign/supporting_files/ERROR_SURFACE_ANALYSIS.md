# Three-Way Error Surface Analysis: Why Meta-TTT Finds Different Optima but the Same Function

A comprehensive weight-space analysis of three training procedures — same-batch
FOMAML, no meta-TTT, and redesigned cross-chunk FOMAML — that land on different
local minima but produce functionally identical models with the same TTT ceiling.

Script: `records/phase3/analysis_three_way.py` (8 analyses, CPU-only, ~3.6s on M2).
Data: `records/phase3/analysis_three_way.json`.

---

## 0. The Puzzle

Three models, trained from the same seed with the same architecture, the same data
order, and the same wallclock budget, differ only in their meta-TTT formulation:

| Model | Meta-TTT variant | legal_ttt | TTT delta |
|---|---|---|---|
| exp101 | FOMAML, same-batch inner/outer | 1.11588 | −0.02342 |
| exp105a | disabled (ablation) | 1.11624 | −0.02331 |
| exp106 | cross-chunk + Δ-loss + MetaSGD | 1.11469* | −0.02299 |

*float-path TTT (int6 canonical crashed due to `meta_sgd` strict-load bug)

**The TTT delta is invariant at ~0.023 bpb.** Three different training objectives —
ranging from "no meta-signal at all" to "theoretically correct cross-document
generalization reward" — produce the same adaptation improvement. This is the
puzzle: why doesn't a better meta-objective produce a better TTT initialization?

The answer lies in the geometry of the loss landscape.

---

## 1. Three Solutions, One Triangle

### 1.1 Weight-space distances form a near-equilateral triangle

The three models are all approximately the same distance from each other:

```
                        exp101
                       /      \
              2335.8  /        \  2356.4        (bank L2 distances)
                     /          \
              exp105a ────────── exp106
                       2324.0
```

| Pair | Bank L2 | Total L2 | Bank cosine |
|---|---|---|---|
| exp101 vs exp105a | 2335.8 | 3312.4 | 0.049 |
| exp101 vs exp106 | 2356.4 | 3345.5 | 0.050 |
| exp105a vs exp106 | 2324.0 | 3237.9 | 0.069 |

**Comment**: The near-equilateral shape means meta-TTT doesn't push you in a
*consistent direction* away from the no-meta solution. Same-batch FOMAML (exp101)
and cross-chunk FOMAML (exp106) are just as far from each other as either is from
no-meta (exp105a). This rules out the hypothesis that meta-TTT is finding a
"meta-optimal" region of weight space — it's finding a *random* neighboring basin,
and the specific basin depends on the exact formulation of the meta-gradient.

### 1.2 Element-wise weight cosine: near-orthogonal everywhere

All three pairs show bank cosines of 0.05–0.07, meaning the raw weight matrices
are effectively orthogonal:

| Pair | qo_bank cos | kv_bank cos | mlp_up cos | mlp_down cos |
|---|---|---|---|---|
| 101 vs 105a | 0.069 | 0.096 | 0.072 | 0.051 |
| 101 vs 106 | 0.063 | 0.075 | 0.074 | 0.050 |
| 105a vs 106 | 0.088 | 0.105 | 0.096 | 0.069 |

**Comment**: These numbers are far below what you'd expect from a 3% training
perturbation on a normally-trained model (where cosine might drop by 0.01-0.02).
The explanation is the Muon optimizer: its Newton-Schulz gradient orthogonalization
amplifies any small perturbation into a large basis rotation. A 3% compute
difference in the gradient (from meta-TTT) compounds across 7000 steps into a
full 90-degree rotation. But the *function* computed by the network depends on
the subspace span, not the basis within it — which brings us to the next analysis.

### 1.3 Scalar control parameters: highly conserved

In contrast to the bank matrices, the per-block control scalars (attn_scale,
mlp_scale, q_gain, resid_mix) are nearly identical across all three models:

| Pair | Scalar avg cosine |
|---|---|
| 101 vs 105a | 0.913 |
| 101 vs 106 | 0.912 |
| 105a vs 106 | 0.927 |

**Comment**: The *macro architecture* of the network — how much attention vs MLP
vs residual each block uses — converges to the same fixed point regardless of
meta-TTT. The scalars that control information flow are not in a degenerate
subspace; they have a single optimum and all three runs find it. Only the internal
*basis* of each weight matrix is free to rotate.

---

## 2. Subspace Overlap: Different Bases, Partially Shared Functions

The principal-angle analysis is the key to resolving the paradox of "orthogonal
weights but identical outputs." We compute the cosines of the principal angles
between the top-k left singular vector subspaces of each weight matrix pair.

### 2.1 Average subspace cosine

| Pair | Avg subspace cosine | Frac dims aligned (>0.9) |
|---|---|---|
| exp101 vs exp105a | 0.615 | 0.411 |
| exp101 vs exp106 | 0.659 | 0.472 |
| **exp105a vs exp106** | **0.727** | **0.548** |

**Comment — the most striking finding**: The no-meta model (exp105a) and the
redesigned meta-TTT model (exp106) share **more** functional subspace than either
shares with the original same-batch FOMAML (exp101). This is counterintuitive:
cross-chunk FOMAML + Δ-loss (the most complex meta-objective) produces a solution
*closer* to vanilla training than the simpler same-batch FOMAML does.

**Interpretation**: Same-batch FOMAML's meta-gradient is systematically biased —
it rewards banks that resist SGD on seen data, which pushes the subspace in a
specific (wrong) direction. Cross-chunk FOMAML's meta-gradient is more like noise
(it's measuring generalization to different documents, which is harder to exploit),
so it perturbs the subspace less than the biased same-batch variant.

### 2.2 Per-matrix subspace overlap

The matrices tell different stories about which functional components are
conserved vs rotated:

**MLP down bank (most stable — all pairs aligned):**

| Pair | Subspace cosine | Frac aligned |
|---|---|---|
| 101 vs 105a | 0.959 | 1.000 |
| 101 vs 106 | 0.969 | 1.000 |
| 105a vs 106 | 0.968 | 1.000 |

**Comment**: The output projection of the MLP is essentially the same function
in all three models. Every principal direction is aligned. This makes physical
sense: `mlp_down` maps from the "concept space" to the "residual stream," and
there's only one good way to do this for a given vocabulary and task.

**MLP up bank (most sensitive to meta-TTT variant):**

| Pair | Subspace cosine | Frac aligned |
|---|---|---|
| 101 vs 105a | 0.551 | 0.500 |
| 101 vs 106 | 0.579 | 0.500 |
| **105a vs 106** | **0.949** | **1.000** |

**Comment**: This is the clearest signal in the dataset. The MLP input projection
(`mlp_up`) is **almost perfectly aligned** between the no-meta and cross-chunk
models, but only ~55% aligned with the same-batch FOMAML model. Same-batch FOMAML
rotated the MLP input subspace away from the natural optimum. The cross-chunk
variant did not — its meta-gradient was too noisy/unbiased to drive a systematic
rotation.

This is direct evidence that same-batch FOMAML's objective mismatch (adapt on
seen data, evaluate on seen data) introduces a *systematic directional bias* into
the MLP's learned feature extraction, while the cross-chunk variant's objective
(adapt on chunk A, evaluate on chunk B) does not.

**KV bank:**

| Pair | Subspace cosine | Frac aligned |
|---|---|---|
| 101 vs 105a | 0.788 | 0.600 |
| 101 vs 106 | 0.807 | 0.800 |
| 105a vs 106 | 0.822 | 0.800 |

**Comment**: The key/value projections show moderate alignment across all pairs,
with the 105a-106 pair again being the most aligned. The attention mechanism's
learned features are partially conserved regardless of meta-TTT, consistent with
the idea that "what to attend to" is well-determined by the task.

**Bigram embedding (most divergent in all pairs):**

| Pair | Subspace cosine | Frac aligned |
|---|---|---|
| 101 vs 105a | 0.213 | 0.000 |
| 101 vs 106 | 0.218 | 0.000 |
| 105a vs 106 | 0.392 | 0.000 |

**Comment**: The bigram table has essentially zero subspace alignment across all
pairs. Zero dimensions are within the 0.9 threshold. This is expected: the
bigram is a low-rank hash table that receives gradient from every forward pass,
so any perturbation to the training signal (meta-TTT or not) creates a completely
different hash embedding. Fortunately, the bigram is a small contributor to the
total model output (learned scale ~0.11), so its divergence has minimal functional
impact.

---

## 3. Error Surface Geometry: Why TTT Sees the Same Landscape from Every Minimum

This is the central question: the three models sit at different points in weight
space, but TTT improves all of them by exactly ~0.023 bpb. What property of the
loss landscape makes this possible?

### 3.1 The two loss surfaces

There are two distinct loss surfaces in play:

```
TRAINING loss surface L_train(θ)          TTT adaptation surface L_ttt(θ, δ)
┌──────────────────────────────┐          ┌──────────────────────────────┐
│                              │          │                              │
│  θ₁₀₁ ●     ● θ₁₀₆         │          │  Same local curvature        │
│        \   /                 │          │  around all three θ          │
│         ● θ₁₀₅              │          │                              │
│    (equivalent minima,       │          │  TTT takes 4 SGD steps       │
│     ~3200 L2 apart)         │          │  along δ from each θ         │
│                              │          │  and gains ~0.023 bpb        │
│  L(θ₁₀₁) ≈ L(θ₁₀₅) ≈ L(θ₁₀₆)│       │  regardless of starting θ    │
└──────────────────────────────┘          └──────────────────────────────┘
```

The training surface `L_train(θ)` has many equivalent minima — the three models
are proof of this. But the TTT surface `L_ttt(θ, δ)`, which measures how much
a few SGD steps on the bank parameters `δ` can reduce the loss on a test chunk,
has **the same curvature at all three minima**.

### 3.2 Bank-level curvature is invariant

The bank weight matrices (qo, kv, mlp_up, mlp_down) are the parameters that
TTT adapts at eval time. Their spectral properties determine how much SGD can
improve the loss in a few steps:

| Property | exp101 | exp105a | exp106 | Interpretation |
|---|---|---|---|---|
| **Condition number** | | | | |
| qo_bank | 1.29 | 1.30 | 1.31 | Near-isotropic — SGD works equally well in all directions |
| kv_bank | 1.32 | 1.38 | 1.38 | Slightly more anisotropic, but identical across models |
| mlp_up_bank | 1.05 | 1.04 | 1.05 | Nearly perfectly conditioned |
| mlp_down_bank | 1.03 | 1.03 | 1.04 | Nearly perfectly conditioned |
| **Effective rank** | | | | |
| qo_bank | 22.0 | 22.0 | 22.0 | All 22 singular directions contribute equally |
| kv_bank | 22.0 | 22.0 | 22.0 | Same — no dimension collapsed |
| mlp_up | 11.0 | 11.0 | 11.0 | Exactly matches the 11-layer bank structure |
| mlp_down | 11.0 | 11.0 | 11.0 | Same |
| **Top-5 energy fraction** | | | | |
| qo_bank | 0.259 | 0.256 | 0.259 | 26% of energy in top 5 of 22 dims — uniform |
| kv_bank | 0.265 | 0.262 | 0.264 | Same |
| mlp_up | 0.467 | 0.466 | 0.465 | 47% of energy in top 5 of 11 dims — near-uniform |
| mlp_down | 0.465 | 0.465 | 0.467 | Same |

**Comment**: Every curvature metric that determines TTT effectiveness is
**identical to 2-3 significant figures** across all three models.

The condition numbers are remarkably low (1.03–1.38), meaning the bank weight
matrices are nearly isotropic — SGD can make equal progress in every direction.
The effective ranks exactly match the structural dimensionality (22 for attention
banks with 2×11 layers, 11 for MLP banks). The energy distribution is near-uniform.

This explains the TTT invariance: when SGD takes 4 epochs of steps on these banks,
it faces the same curvature landscape regardless of which training minimum the
model started from. The ~0.023 bpb gain is determined by the **TTT optimizer
configuration** (SGD with momentum 0.9, cosine LR, 4 epochs, 65K-token chunks)
operating on a **near-isotropic** bank parameter space — not by the initialization
quality.

### 3.3 The one difference: spectral gap

The spectral gap (σ₁ − σ₂) is the only bank-level metric that differs
meaningfully between models:

| Bank | exp101 | exp105a | exp106 |
|---|---|---|---|
| qo_bank | 0.294 | 0.377 | 0.483 |
| kv_bank | 0.380 | 0.336 | **1.169** |
| mlp_up_bank | 0.607 | 0.119 | **1.520** |
| mlp_down_bank | 0.275 | 0.226 | 0.310 |

**Comment**: exp106's kv_bank and mlp_up_bank have spectral gaps 3-12x larger
than the other two models. This means the dominant singular value is more
"peaked" relative to the second — the weight matrix has a stronger directional
preference.

This is likely an artifact of the cross-chunk split: when the inner loop adapts
on different documents than the outer loop evaluates on, the meta-gradient has
a component that aligns the dominant singular direction with cross-document
generalizable features. But this alignment doesn't translate into a larger TTT
delta, because the condition number (which determines SGD's progress) remains
the same — the gap grows while the overall spectrum stays isotropic.

In other words: exp106 learned a slightly more "opinionated" first singular
direction, but TTT doesn't care about the first direction specifically — it
moves the banks along all directions equally.

---

## 4. Mode Connectivity: Distinct Basins, Neighboring Landscapes

### 4.1 Pairwise midpoint analysis

If two models are in the same loss basin, their midpoint (average of weights)
should have similar norm to either endpoint. Norm collapse indicates vector
cancellation, which means the two models' weight matrices are pointing in
different directions — characteristic of different basins.

| Pair | L2 distance | Midpoint norm ratio | Basin assessment |
|---|---|---|---|
| exp101 vs exp105a | 3312.4 | 0.786 | Different basins |
| exp101 vs exp106 | 3345.5 | 0.793 | Different basins |
| **exp105a vs exp106** | **3237.9** | **0.807** | **Borderline same basin** |
| 3-way centroid | — | 0.704 | Clearly distinct |

**Comment**: The threshold for "same basin" is roughly 0.8. exp105a and exp106
(no-meta and cross-chunk meta) are right at the boundary — they might be in the
same broad basin or in very close neighboring basins. exp101 (same-batch FOMAML)
is clearly in a different basin from both.

This is consistent with the subspace overlap findings: same-batch FOMAML pushes
the model furthest from the natural optimum, while cross-chunk FOMAML stays
closer to where vanilla training would have landed.

### 4.2 Centroid analysis

The centroid (average of all three models) has a norm ratio of 0.704 — a 30%
norm loss from vector cancellation. This confirms the three models are genuinely
in different regions of weight space, not just slightly shifted versions of the
same solution.

```
  Individual model norms:    ~2900 (each)
  Centroid norm:             ~2042
  Norm loss:                 ~30%
```

If all three were in the same basin, the centroid would have norm ~2900. The 30%
deficit means the three weight vectors are canceling each other — like averaging
three unit vectors pointing in different directions.

---

## 5. Quantization Sensitivity: The Surface is Flat

| Model | Avg int6 MSE | Relative to exp101 |
|---|---|---|
| exp101 | 8.686 × 10⁻⁵ | baseline |
| exp105a | 8.691 × 10⁻⁵ | +0.06% |
| exp106 | 8.686 × 10⁻⁵ | 0.00% |

**Comment**: The quantization error surface is flat across all three minima.
Per-row int6 quantization with GPTQ-style Hessian-informed column ordering
adapts its scales to whatever weight distribution it finds. The per-row amax
adjusts to the local weight range at each minimum, so the roundtrip MSE is
independent of which minimum the model occupies.

This rules out the hypothesis that meta-TTT could serve as an implicit
quantization-aware regularizer. It cannot — the quantization pipeline's per-row
adaptation is more powerful than anything meta-TTT does to the weight distribution.

---

## 6. MetaSGD Scale Convergence

The 66 MetaSGD parameters (`meta_sgd_{qo,kv,up,down}`, one per bank-type per
layer) were excluded from `final_model.pt` by the export filter, so they cannot
be analyzed from the saved checkpoint. Their convergence behavior was observed
during training:

- All 66 scales converged to values **near 1.0** (their initialization)
- No meaningful per-layer differentiation was learned
- Standard deviation across all 66 scales was <0.04

**Comment**: The MetaSGD result is a "dog that didn't bark." If the meta-training
signal were strong enough to learn useful per-layer adaptation speeds, we'd see
some layers with scales > 1 (adapt faster) and others with scales < 1 (adapt
slower). Instead, uniform convergence means the meta-gradient's per-layer
component is below the noise floor of the optimizer.

At `META_TTT_EVERY=4` (one meta-step per 4 training steps), the meta-gradient
contributes ~25% of gradient updates but at only ~30% of the main gradient's
magnitude (due to `META_TTT_LOSS_WEIGHT=0.5` and the Δ-loss dilution). The
effective meta-signal is ~7.5% of total gradient energy — too weak to drive 66
scalar parameters away from their initialization over 6686 training steps.

---

## 7. The Big Picture: Why Meta-TTT Cannot Move the TTT Ceiling

### 7.1 The argument from curvature invariance

The TTT delta depends on:

1. **How far SGD can move the banks** in 4 epochs — determined by the learning
   rate, momentum, and number of steps (fixed across all experiments)
2. **How much loss reduction each step buys** — determined by the local curvature
   of the loss surface around the bank parameters

We showed (Section 3.2) that the local curvature is identical at all three
minima: condition numbers 1.03–1.38, effective ranks exactly matching the
structural dimensionality, energy distributions near-uniform. SGD makes the
same progress per step from any of the three starting points.

### 7.2 The argument from over-parameterization

The training loss surface has a degenerate set of equivalent minima (the three
models are proof). Over-parameterization theory tells us that gradient-based
optimization in this regime converges to *any* minimum in the connected set,
depending on the optimization trajectory. The meta-TTT gradient perturbs the
trajectory, selecting a different minimum — but all minima in the set have the
same loss, the same local curvature, and the same TTT adaptation potential.

Meta-TTT would help only if it could find a minimum *outside* the connected
set — one with different curvature properties that make SGD more effective.
But with first-order MAML and a single inner step, the meta-gradient is too
similar to the regular training gradient to escape the set. It's a perturbation
within the basin, not a jump to a different landscape.

### 7.3 The argument from the spectral gap exception

The one metric that DID differ was exp106's spectral gap (Section 3.3). The
cross-chunk meta-gradient successfully created a more "peaked" dominant singular
direction in kv_bank and mlp_up_bank. But this didn't help TTT because:

- TTT uses SGD with momentum, which converges based on the *worst* direction
  (condition number), not the *best* direction (dominant SV)
- The condition number (ratio of largest to smallest SV) stayed the same
- Making the top SV more peaked doesn't help if the bottom SVs are unchanged

To move the TTT ceiling, you'd need to change the *shape* of the SV spectrum —
make all directions better, or specifically improve the worst directions. A
stronger meta-training signal (second-order MAML, more inner steps, dedicated
meta-training phase) might achieve this, but first-order MAML with one inner
step fundamentally cannot.

### 7.4 The architecture-limited ceiling

The ~0.023 bpb TTT delta is set by:

- **Bank dimensionality**: 4 bank types × 11 layers, each a (d, d) or (d, kv_d)
  matrix. This is the number of free parameters TTT can adapt.
- **TTT data**: 947 chunks × 65K tokens each. This is how much test-time evidence
  is available for adaptation.
- **TTT optimizer**: SGD with momentum 0.9, cosine LR schedule, 4 epochs.

None of these depend on the training-time meta-objective. The ceiling is a
property of the architecture × TTT-optimizer interaction, not the training
procedure.

**To raise the ceiling**, you'd need to change one of these:
- More adaptable parameters (more bank layers, rank-1 correctors, LoRA-style)
- Better TTT optimizer (Adam, higher LR, more epochs)
- More test-time data (larger chunks, more chunks)
- Different bank structure (allow cross-layer adaptation)

---

## 8. Reproducing This Analysis

```bash
# From the repo root:
python3 records/phase3/analysis_three_way.py
```

Runtime: ~3.6 seconds on Apple M2 (CPU only, no GPU needed).

Required checkpoints:
- `records/phase3/exp101_poscond-bigram-trigram_from_exp95/_pod/final_model.pt`
- `records/phase3/exp105a_no-metattt_from_exp101/_pod/final_model.pt`
- `records/phase3/exp106_metasgd-crosschunk-delta_from_exp101/_pod/final_model.pt`

Output:
- `records/phase3/analysis_three_way.json` (full numerical results)
- Executive summary to stdout

---

## 9. Summary of Findings

| Finding | Evidence | Section |
|---|---|---|
| Three models form a near-equilateral triangle in weight space | Bank L2 distances: 2324–2356 | 1.1 |
| Bank weights are near-orthogonal element-wise (cos ~0.05–0.07) | Muon amplifies small gradient perturbations into full basis rotations | 1.2 |
| Macro network structure is conserved (scalar cos ~0.91–0.93) | attn_scale, mlp_scale, q_gain converge to same fixed point | 1.3 |
| Cross-chunk FOMAML (exp106) is closer in subspace to no-meta (exp105a) than same-batch FOMAML (exp101) is | Subspace cosine: 105a-106 = 0.727 vs 101-105a = 0.615 | 2.1 |
| mlp_up_bank: same-batch FOMAML rotates the subspace; cross-chunk does not | 105a-106 cos = 0.949 vs 101-105a cos = 0.551 | 2.2 |
| Bank curvature (condition number, effective rank, energy distribution) is identical across all three models | Cond 1.03–1.38, eff_rank = 22/11, top5_energy = 0.26/0.47 | 3.2 |
| exp106 has larger spectral gaps in kv_bank and mlp_up_bank | kv: 1.169 vs 0.38; up: 1.520 vs 0.12–0.61 | 3.3 |
| exp105a and exp106 are borderline in the same basin; exp101 is in a different basin | Midpoint ratios: 0.807 vs 0.786 | 4.1 |
| Quantization sensitivity is identical | MSE range: 8.686–8.691 × 10⁻⁵ | 5 |
| MetaSGD scales converged to uniform ~1.0 | No per-layer LR differentiation learned | 6 |
| TTT ceiling is architecture-limited, not init-limited | Curvature invariance + over-parameterization argument | 7 |

---

## TL;DR

Three meta-TTT formulations — same-batch FOMAML, no meta-TTT, and cross-chunk
FOMAML with Δ-loss + MetaSGD — find three distinct local minima in weight space
(equilateral triangle, ~2300 L2 apart, bank cosine ~0.06). But these minima have
**identical local curvature** (condition numbers 1.03–1.38, effective ranks exactly
matching layer count, energy distributions near-uniform), which is why TTT improves
all three by the same ~0.023 bpb. The loss landscape is degenerate: many equivalent
minima exist, meta-TTT selects which one you land in, but the TTT adaptation
surface looks the same from every minimum. The ceiling is set by the bank
dimensionality and TTT optimizer, not by initialization quality. The one surprising
finding: same-batch FOMAML systematically rotates the MLP input subspace (cos 0.55
with no-meta), while cross-chunk FOMAML preserves it (cos 0.95) — the biased
meta-objective produces a more disruptive (but not more useful) perturbation.
