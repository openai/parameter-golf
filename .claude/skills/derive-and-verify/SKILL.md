---
name: derive-and-verify
description: Invoke when about to derive an SSM equation, kernel formula, discretization, or non-default initialization in scratch/ — before writing the train_gpt.py implementation. ALSO invoke when an experiment's *interpretation* depends on a specific harness/code mechanism (e.g. quantization protection, tensor-name patterns, gradient masking) — read the relevant code path end-to-end before launching, because the discipline ("does the math/code I'm assuming actually exist?") is the same. Carries the patterns for math-heavy research code (worked tiny example, cite reference formula, recurrence-vs-convolution as free oracle, init invariants, degenerate cases, print spectra) so silent bugs surface upstream rather than as a step-50 NaN. Especially apt for SSMs where representations mix and the recurrence amplifies errors. Distinct from pull-out (which is mode-shift, not how-to-math).
---

# Derive and Verify

The SSM literature mixes representations (continuous vs discrete, real vs complex, recurrence vs convolution) and the recurrence amplifies math errors over the sequence length. Silent bugs are common. This is the discipline you bring to `scratch/` before writing the implementation — patterns to adapt, not a procedure to follow.

**The cheapest debugging is the kind you do before training.** The recurrence will surface mistakes downstream as a NaN at step 50 or a worse-than-baseline val_bpb — but those mistakes were already visible upstream; you just hadn't looked.

## When to invoke
- Before deriving any new SSM equation in `scratch/` (discretization, kernel formula, selective scan, etc.)
- Before writing a custom selective-scan, kernel constructor, or non-default initialization
- Before initializing parameters with a specific formula (HiPPO-LegS, A_log, dt_bias) where getting the formula wrong is a silent failure
- After reading a primer section that introduces a new equation you'll implement
- **Before any experiment whose *interpretation* depends on a specific harness or code path** — e.g. "split in_proj for fp32 protection" presumes `CONTROL_TENSOR_NAME_PATTERNS` affects training (it doesn't — it's serialization-only); "freeze the BG path" presumes a particular gradient mask exists. Five minutes of reading the code path beats running an experiment whose result you can't interpret. The discipline is the same as for math: confirm the mechanism you're assuming actually exists before leaning on it.

## When NOT to invoke
- Routine env-var tweaks
- Reading existing code (use `search_journal` or just `Read`)
- Trajectory-pattern checks during a run (`launch-and-await` covers that)
- Re-using a previously-verified block in a new experiment

## The patterns

### 1. Worked tiny example first
B=1, L=4, D=2, N=2. Compute by hand on paper. Compare to your code element by element. If you can't compute it by hand, you don't understand the operation well enough to debug it later. This is not optional — it is the difference between deriving and guessing.

### 2. Cite the reference formula in code
Every key equation gets a comment with the source:
```python
# ZOH per primer §1.2: Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I) Δ B
```
The cite is the version-controlled source of truth; your code is the implementation. Future-you (or a subagent) verifies code against intent via the cite.

### 3. Free oracle: recurrence-vs-convolution duality
For LTI blocks (S4, S4D), given the same `(A, B, C, Δ)`: the recurrent rollout `x_k = Ā x_{k-1} + B̄ u_k` and the convolution `y = K * u` must agree on the same input. If they don't, one of your implementations is wrong. This is the cheapest sanity check available for any S4-family block you write.

For selective (Mamba-family) scans, use `references/selective_scan_ref.py` as the oracle — see `references/INDEX.md` for the protocol.

### 4. Print invariants on init (once, at construction)
- **Eigenvalues of Ā in the closed unit disk**: `torch.linalg.eigvals(A_bar).abs().max()` should be ≤ 1. The standard `A = -exp(A_log)` parameterization gives this for free post-discretization, but verify.
- **dt strictly positive**: if `dt = softplus(dt_proj(x) + dt_bias)`, print `dt.min()` to confirm > 0. A negative dt silently breaks the recurrence's stability properties.
- **Kernel decay** (for LTI): `K[-1].abs().mean()` should be small if your timescales are right. If `K[-1] ≈ K[0]`, your timescales are too long for the sequence length.

### 5. Closed-form degenerate cases
- A=0 → pass-through: y = D*u (or zero if D=0)
- B=0 → output stays at zero regardless of input
- Δ=0 → no state updates, output collapses to D*u
- C=0 → output is zero regardless of state

Set the parameter, run a forward pass, confirm the output matches the closed form. Quick "is the wiring right" checks before training.

### 6. Print spectra at first forward (then remove)
During the derive-and-implement phase only — not in production training loops. Print:
- Eigenvalue spectrum of Ā (check stability, check distribution)
- dt distribution (`dt.histogram(...)` or just min/max/mean)
- A̅ row norms (catches outlier states)
- Kernel decay shape for LTI (early indication of timescale match)

These tell you whether the math you derived is the math the code is computing. Remove the prints once you've confirmed agreement; you don't want them firing every forward in a 200-step run.

## After
With patterns satisfied, the implementation is ready for the experiment loop. Trajectory still has to look right (step 1 ≈ ln(vocab), monotonic descent — see `launch-and-await`), but a clean step-1-to-10 trajectory plus pre-training derivations together is much stronger evidence of correctness than either alone. For SSMs especially, see also program.md "SSM-specific harness facts" → late-NaN gate (step-100 await is non-optional).

## If a derivation fails
If the duality doesn't agree, an eigenvalue is outside the unit disk, dt prints negative, or the worked tiny example mismatches: **do not run the experiment.** Fix the math first. Update `scratch/` with what you found wrong. The cost of fixing math in `scratch/` is minutes; the cost of fixing it via training-time debugging is hours.

## Distinct from pull-out
| Skill | Purpose | When |
|---|---|---|
| `pull-out` | Mode-shift to higher-level reassessment | Every reflective transition |
| `derive-and-verify` (this skill) | Patterns for *how* to do math well in `scratch/` | When about to derive an equation |

Pull-out tells you to use scratch/ ("compute the parameter count, sketch the math"). This skill tells you *how* to do that well for SSM-flavored math.
