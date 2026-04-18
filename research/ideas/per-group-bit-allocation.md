# Per-group quantization bit allocation

**Status:** candidate (speculative — proposed but not shipped by any submission)
**Expected Δ:** unknown; plausibly +0.0000 to +0.0015 bpb. High variance.
**Source:** Two independent lines of reasoning converge on this idea:
1. `records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/README.md`, "Future Directions" — author's empirical observation that per-group Hessian traces are stable across seeds with ~30× spread.
2. General mixed-precision-quantization literature — sensitive matrices (attention QK, early layers) tend to dominate quant error and benefit from extra bits.

## Idea
SOTA uses uniform INT6 for all matrix weights and INT8 for embeddings. Allocate bits asymmetrically within the same 16MB budget, spending more bits where quant error costs the most loss and fewer bits where matrices compress well anyway.

**Two grouping axes** to consider (possibly combine):

### Axis 1 — by layer depth (Hessian-driven)
Near-SOTA diagnostics: group-level Hessian traces are **very stable across seeds (r=0.997)** with a **~30× hierarchy**: early blocks >> loop blocks >> mid >> late.

Suggested groups for current SOTA (11 layers, loop 3-5, parallel residuals 7+):
- **Early**: layers 0, 1, 2 → INT7 or INT8 candidate
- **Loop**: layers 3, 4, 5 → INT6 (baseline)
- **Mid**: layer 6 → INT6
- **Late**: layers 7, 8, 9, 10 → INT5 candidate

### Axis 2 — by matrix type within a layer
Different matrix types have different Hessian and redundancy profiles:
- **Attention QK**: narrow but highly sensitive → INT7 candidate
- **MLP up-projection**: often highly redundant / low-rank → INT5 candidate
- **Attention V / output / MLP down**: baseline → INT6

Axis 1 and 2 compose: you can allocate per-(layer-group × matrix-type) for finer control, at the cost of more search-space complexity.

## Why it might help
- Hessian trace = "how much does loss change if I perturb this matrix." Groups with 30× more trace contribute 30× more to quant error per unit perturbation.
- GPTQ's total error is dominated by a few high-sensitivity matrices; an extra bit there often wins back 0.001-0.003 bpb in mixed-precision literature.
- Brotli compression benefits from favorable entropy distribution. INT5 on a low-entropy matrix can compress better post-Brotli than uniform INT6.
- **Per-group, not per-row**: the r=0.997 signal stability means allocation generalizes across seeds. Per-row importance (r=0.12 in the same diagnostics) is too noisy to use.
- Free headroom: if late blocks truly tolerate INT5, the freed bytes could pay for a future feature addition instead of improving bpb. Either way, it's budget we're currently not spending optimally.

## Hotstart screening plan
**Pure quant-time change. Zero training recompute.**

- **Hotstart from:** `runs/000-sota-replication/checkpoints/ckpt_final_pre_ema_step~4550.pt`.
- **Pipeline:** EMA → GPTQ with per-group bit allocation → Brotli → eval.
- **Control:** standard uniform INT6.
- **Wall per trial:** 2-3 min.
- **Cost per trial:** <$0.50.
- **Promotion threshold:** Δ ≥ 0.0005 over uniform INT6 control. Below that the complexity isn't worth it.

### Suggested initial trials (don't search the full combinatorial space)
Hand-pick 4-6 allocations motivated by the two axes:

| Trial | Allocation | Axis | Hypothesis |
|---|---|---|---|
| T0 | uniform INT6 | — | control |
| T1 | early=INT7, rest=INT6 | layer-depth | spend where Hessian is highest |
| T2 | early=INT7, late=INT5 | layer-depth | balance: bits moved from late to early |
| T3 | QK=INT7, MLP-up=INT5, rest=INT6 | matrix-type | type-based sensitivity |
| T4 | early=INT7 + QK=INT7, MLP-up=INT5 | combined | both axes stacked |
| T5 | whatever T1/T2/T3 showed + tuning | refinement | |

## Budget math (the gating question)
11 layers × multiple matrices × 512² = roughly 30MB of weight storage before quantization. INT6 brings it into the 16MB range after Brotli. Per-bit delta matters.

Rough heuristic for a single 512×512 matrix:
- INT4: 128 KB
- INT5: 160 KB
- INT6: 192 KB (baseline)
- INT7: 224 KB (+16% over INT6)
- INT8: 256 KB (+33% over INT6)

Moving one matrix from INT6 → INT7 costs ~32 KB pre-Brotli. The 16MB budget is post-Brotli — Brotli typically compresses 20-40% more, so effective overhead is smaller but variable.

**The blocker:** without a precise accounting of post-Brotli sizes per matrix, it's easy to produce a disqualified submission. First implementation task should be a **size-verifier** that, given a bit-allocation config, reports whether the post-Brotli artifact will fit in 16MB *before* the expensive GPTQ+Brotli run.

## Code-change sketch
- Define layer groups (default: `[[0,1,2], [3,4,5], [6], [7,8,9,10]]`).
- Optional: matrix-type classifier (default: treat all matrices in a layer equivalently).
- Per-group-and-type `n_bits` parameter (default 6 everywhere = baseline).
- GPTQ quant routine accepts per-matrix n_bits.
- **Required addition:** a size-verifier that predicts post-Brotli size from bit allocation; gate any trial on this passing.

Env interface (example): `BIT_ALLOC_LAYERS="6,6,6,6,6,6,6,6,6,6,6"` or richer JSON spec.

## Risks / open questions
- **16MB budget enforcement.** Main blocker. Need a verifier before the first run.
- **Group boundaries.** The "early/loop/mid/late" split is a hypothesis, not a measurement. Running the actual Hessian-trace analysis on spec-000 checkpoints (cheap: part of the normal GPTQ calibration) would sharpen the grouping.
- **GPTQ behavior at edges.** GPTQ's column-wise error correction may degrade at INT4 (ternary weights shipped on 2026-03-24 but weren't competitive). INT5 is probably the safe lower bound.
- **Embed and head layers.** Already special-cased in SOTA. Don't include in the reallocation without a specific reason.
- **Interaction with Hessian-SDClip (Candidate 1).** Both modify the quant pipeline. If both screen well individually, test them stacked in a single screen — not additive by assumption.
- **Calibration data per bit width.** GPTQ uses the same activation statistics regardless of bit width. Fine at INT6; at INT4 the linearity assumption behind error correction likely breaks.

## If this works
- Novel contribution — no submission has shipped per-group bit allocation.
- Stacks with Hessian-SDClip after joint testing.
- Freed bits in late blocks could alternatively be spent on a future feature (wider model, extra scalar params, etc.) rather than bpb directly. That's a strategic option for stacking with a training-time change.

## Relationship to other ideas
- **Supersedes `per-group-quant.md`** — that file was the original sketch of this idea (matrix-type axis only). Consolidated here.
- **Orthogonal to Progressive Recurrence** (training-time change).
- **Adjacent to Hessian-SDClip** (both quant-pipeline). If both screen well, they must be tested stacked.
