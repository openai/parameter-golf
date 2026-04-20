# Rotation regime-dependence — helps long context, hurts short context

**Status:** 🟡 OPEN — strong empirical signal, active exploration via spec 010b (site-selective ablation). Decision point after 010b's results.
**Observed Δ (spec 010 port_1695 vs spec 009 baseline, same 8×H100 eval):**
- Longest docs (dl 1320–10426): **−0.0064 bpb**
- First 40 long docs (dl 730–10426): **−0.0073 bpb**
- Middle docs (dl 437–712): **+0.0088 bpb**
- Shortest docs (dl 83–263): **+0.0146 bpb**
- Aggregate over full eval: **−0.00005 bpb** (null by cancellation)

**Source:** Per-batch analysis of spec 010's `ttp:` log. The sign flip around doc-length ~500 tokens is reproducible and follows a clean monotonic curve with doc length.

## The phenomenon

Applying Hadamard rotations to weight/activation pairs before GPTQ (as in #1695's SpinQuant V1 port) produces a **regime-dependent change in model quality**:

- Model becomes **better** on long-context prediction (~−0.007 bpb on docs > 1000 tokens).
- Model becomes **worse** on short-context prediction (~+0.015 bpb on docs < 300 tokens).
- Crossover is around 400–700 tokens of document length.
- Float forward pass is mathematically identical to unrotated (orthogonal rotations cancel: `F.linear(x @ R, W @ R) = F.linear(x, W)`). Only the *quantized* model differs.

The effect exists **at both pre-TTT and post-TTT eval stages**. `diagnostic_quantized` Δ was −0.00009; `quantized_ttt_phased` Δ was −0.00005. TTT doesn't absorb the rotation effect — it preserves it (equalizing all variants down by the same ~0.013 from TTT's own contribution, but keeping the relative rotation signal intact).

## Why it happens (working hypothesis)

Rotation spreads quantization error more evenly across activation dimensions:

- **Long contexts** aggregate information across many tokens. Per-token quant errors average out over the sequence; the *mean* error is what matters. Rotation lowers the mean → lower loss on long-context predictions.
- **Short contexts** have almost no aggregation. Per-token prediction depends on just a few recent tokens. Per-token *variance* matters more than mean. Hadamard rotation adds small per-token perturbations (signed ±1/√d mixing + bf16 roundoff), and on short docs those aren't averaged out.

This is a physical explanation, not a proven one. Spec 010b's site-ablation is the first empirical test of the "which sites carry which half" sub-question.

## Why it's interesting

**In principle**, the decomposition into regime-specific effects gives us a lever we didn't have before. Two cases where this could land a net-positive on parameter-golf:

1. **Site-selective rotation** (spec 010b): if attention rotations carry the long-context benefit and MLP rotations carry the short-context hurt, running `attn_only` mode delivers asymmetric benefit that might net positive against the eval distribution.
2. **Layer-selective rotation** (potential spec 010c): rotate only decoder layers (6–10), skip encoder (0–5). Decoder aggregates context; encoder handles local features. Same hypothesis at a different granularity.
3. **Rotation-aware retraining** (spec 011+ territory): train weights that are robust to rotation. Could eliminate the short-doc hurt while keeping the long-doc help.

**Out of scope for exploitation** (due to the 16 MB artifact cap):
- Ship two quantized models (one rotated, one not) — doesn't fit.
- Toggle rotation per doc on a single model — GPTQ quantization is basis-specific, so dequantized weights used in the "wrong" basis incur extra quant error that drowns the benefit.
- Per-position rotation modulation during inference — breaks float invariance for uncertain gain.

## Why it's an idea, not yet a plan

The phenomenon is real and documented. The **exploitation** is what's open:

- Spec 010b is the first exploitation attempt. Four modes, one pod session, ~$25. If `attn_only` lands, we have a measured lever.
- If 010b is null across all site partitions, the regime-dependence doesn't decompose by site and the only remaining exploitation path is rotation-aware retraining (much more expensive; at least a full retrain, possibly with co-designed model changes).
- If 010b shows a clear `attn_only` win, the natural follow-up is combining it with other levers (tapered WD, novel quant tricks) to see if the decomposition is stable under composition.

## How to test / extend

1. **Site-selective ablation** — spec 010b. Decomposes rotation into {attn_only, mlp_only, all, attn_in_only} to identify which subset carries the net-positive.
2. **Layer-selective ablation** — potential spec 010c. Applies rotation only to layers ≥ 6 (decoder), leaving encoder unrotated. If rotation's help lives in decoder aggregation and hurt lives in encoder feature extraction, this flattens the regime profile.
3. **Seed sweep** — the Hadamard × sign-diag construction is seed-dependent. Different seeds might produce rotations with different regime profiles. Cheap (~$5 per seed).
4. **Per-bucket diagnostic on non-rotation levers** — when measuring anything else (spec 011 tapered WD, for example), we should also bucket by doc length. If tapered WD helps short docs specifically, it could compose with rotation-on-long-only to get the best of both.
5. **Rotation-aware retraining** — train with `_sq_active=True` from step 0, so the weights learn to be robust to rotation. Untested; possibly the cleanest lever if we can afford a retrain at a later stage.
6. **Validate the hypothesis on a simpler model** — reproduce the regime-dependence on a tiny toy transformer (GPT-2-tiny or smaller) to confirm it's a property of Hadamard rotation + softmax/MLP attention, not a quirk of #1736's stack. Research-interest level, not competition-critical.

## What this doesn't tell us

- Whether the regime effect is specific to #1736's stack (parallel residuals + phased TTT + CaseOps) or general to transformers. #1695's base didn't show a clean regime finding — possibly because they didn't bucket by doc length, or possibly because their stack produces a different regime profile.
- Whether the effect survives architectural changes we might make. If we switch from phased TTT to doc-indep TTT (spec 006's old idea), the regime profile likely shifts.
- Whether "crossover at 400–700 tokens" is a property of the rotation or a property of #1736's effective context-use pattern.

## Risks if we try to exploit

- **010b could be null.** Regime-dependence may be smeared uniformly across all 4 sites, and no subset isolates the net-positive.
- **`attn_only` could win but stop working when composed with other levers.** The regime profile might shift after a tapered-WD retrain.
- **Seed dependence.** If the regime profile varies a lot with rotation seed, the finding is fragile — not a reliable lever.

## Links

- Parent: `research/ideas/1736-improvement.md`, `research/ideas/spinquant-integration-notes.md`
- Evidence: `runs/009-spinquant-hotstart/`, `runs/010-port-1695/`, `diary/2026-04-20-spinquant-results-and-regime-finding.md`
- Active follow-up spec: `research/specs/010b-spinquant-sites.md`
