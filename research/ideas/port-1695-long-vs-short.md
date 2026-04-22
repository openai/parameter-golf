# Observation — #1695's online rotation approach favors long sentences over short ones

**Status:** 📝 OBSERVATION from spec 010 run.
**Source:** Per-batch analysis of `runs/010-port-1695/run.log` vs `runs/009-spinquant-hotstart/baseline/run.log`. See also `diary/2026-04-20-spinquant-results-and-regime-finding.md`.

## The observation

PR #1695's online-Hadamard-rotation scheme, when ported to #1736's stack and run as spec 010, produces a **regime-dependent change in prediction quality** keyed to document length.

Per-batch bpb, bucketed by the batch's longest doc length (`dlmax`):

| Doc length bucket | baseline bpb | port_1695 bpb | Δ (port − base) |
|---|---|---|---|
| longest (dl 1320–10426) | 1.0538 | 1.0474 | **−0.0064** |
| first 40 (dl 730–10426) | 1.0544 | 1.0471 | **−0.0073** |
| middle (dl 437–712) | 1.0629 | 1.0718 | **+0.0088** |
| shortest (dl 83–263) | 1.1752 | 1.1898 | **+0.0146** |

Sign flips cleanly around doc length ~500 tokens.

- On **long sentences / long docs**, the rotated-and-quantized model predicts **better** than baseline.
- On **short sentences / short docs**, it predicts **worse**.
- In aggregate over the whole eval distribution, the two cancel to ~0 (Δ = −0.00005 bpb at the reported `val_bpb`).

## Shape of the intra-eval trajectory

Eval processes batches in length-sorted order (longest first). The running-average bpb (`rb` column in the log) reflects this order:

| batches done | baseline rb | port_1695 rb | Δ |
|---|---|---|---|
| 25 | 1.0771 | 1.0604 | −0.017 |
| 116 | 1.0625 | 1.0524 | −0.010 |
| 500 | 1.0601 | ~1.057 | −0.003 |
| 773 | 1.0663 | 1.0604 | −0.006 |
| final | 1.06728 | 1.06723 | −0.00005 |

Looking at the trajectory alone, port_1695 appears to be winning by a wide margin early (up to −0.017 bpb vs baseline at batch 25). That signal is real on the long-doc subset. It erodes as the eval advances into progressively shorter docs where rotation starts hurting, and the aggregate resolves to near-zero.

## Why this probably happens

Working hypothesis (physical, not proven):

- Rotation spreads quantization error more evenly across activation dimensions.
- **Long contexts** aggregate across many tokens → per-token quant errors average out → what matters is the *mean* error, which rotation lowers. So long-context perplexity improves.
- **Short contexts** have almost no aggregation → predictions depend on a few recent tokens → what matters is per-token *variance*, not mean. Rotation's small per-token perturbations (signed-Hadamard mixing + bf16 roundoff) don't average out on short docs, and actively hurt.

## Things this observation tells us

1. **The null aggregate in spec 010 is a cancellation, not an absence of effect.** The rotation is doing something meaningful; it just happens to help and hurt in proportions that net out against our specific eval distribution.
2. **Rotation is regime-specific.** Long-sentence modeling behavior and short-sentence modeling behavior are decoupled by rotation.
3. **Running averages during length-sorted eval are misleading.** Mid-eval `rb` columns favor long-doc behavior and will look optimistic or pessimistic depending on where rotation sits relative to baseline.

## Things this observation does NOT tell us

- Whether the effect is specific to #1695's 4-rotation scheme, or generic to any Hadamard rotation in the SpinQuant family. (Spec 009's R_a-only rotation was null in aggregate but we didn't compute the per-bucket breakdown; could have the same long-vs-short profile at smaller magnitude.)
- Whether the crossover doc-length (~500 tokens) is a property of rotation specifically or of #1736's effective context utilization.
- Whether the effect would compose predictably with other changes (tapered WD, SwiGLU, different TTT configurations).

## Cross-references

- Parent idea: `research/ideas/rotation-regime-dependence.md` (broader framing + exploitation paths)
- Triggering run: `runs/010-port-1695/`
- Baseline for comparison: `runs/009-spinquant-hotstart/baseline/`
- Spec: `research/specs/010-port-1695-online-rotation.md`
- Follow-up spec exploring which rotation sites carry which half: `research/specs/010b-spinquant-sites.md`
